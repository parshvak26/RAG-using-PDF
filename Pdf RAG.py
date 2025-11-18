#!/usr/bin/env python3
"""
rag_pdf_genai.py

Single-file RAG demo:
 - Extract text from a PDF
 - Clean + chunk the text (basic NLP preprocessing)
 - Embed chunks with sentence-transformers
 - Build a simple NearestNeighbors index (scikit-learn)
 - For each user question: clean question, retrieve top-k chunks, construct RAG prompt
 - Call Google GenAI (gemini-2.5-flash) using the provided snippet format
 - Print response

Usage:
    export GOOGLE_API_KEY="your_key_here"
    python rag_pdf_genai.py /path/to/file.pdf

Dependencies:
    pip install PyPDF2 sentence-transformers scikit-learn numpy genai
"""
import os
import sys
import argparse
import math
import re
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import PyPDF2
from google import genai
os.environ["GOOGLE_API_KEY"] = "Your API KEY"

# -------------------------
# Configuration
# -------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast embedding model
CHUNK_SIZE = 800                        # characters per chunk (approx)
CHUNK_OVERLAP = 200                     # characters overlap between chunks
TOP_K = 5                               # number of chunks to retrieve
DEVICE = "cpu"                          # sentence-transformers will autodetect GPU if available

# -------------------------
# Utilities: PDF extraction
# -------------------------
def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    reader = PyPDF2.PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            # some PDFs are weird, don't die on a page
            pages.append("")
    text = "\n".join(pages)
    return text

# -------------------------
# NLP cleaning & chunking
# -------------------------
def clean_text(s: str) -> str:
    """Basic text cleaning for documents:
       - Normalize unicode whitespace
       - Remove repeated whitespace
       - Replace multiple hyphens/newlines
       - Remove weird control characters
    """
    if not s:
        return ""
    # Normalize whitespace
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    # Remove multiple newlines to maximum two
    s = re.sub(r"\n{3,}", "\n\n", s)
    # Remove control chars
    s = "".join(ch for ch in s if ord(ch) >= 9 and ord(ch) <= 0x10FFFF)
    s = s.strip()
    return s

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, int, int]]:
    """Split a long text into overlapping chunks.
       Returns list of tuples (chunk_text, start_char, end_char)
    """
    text = text.strip()
    n = len(text)
    if n == 0:
        return []
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# -------------------------
# Prompt cleaning for user question
# -------------------------
def clean_question(q: str) -> str:
    """Lightweight cleaning of user queries:
       - strip, collapse whitespace
       - ensure it ends with a question mark
       - remove non-printable chars
    """
    if not q:
        return q
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    q = "".join(ch for ch in q if ch.isprintable())
    if not q.endswith("?"):
        q = q + "?"
    return q


class SimpleRAGIndex:
    def __init__(self, embed_model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(embed_model_name)
        self.chunks = []            # list of (text, start, end)
        self.embeddings = None      # numpy array shape (n_chunks, dim)
        self.nn = None              # NearestNeighbors index

    def add_chunks(self, chunks: List[Tuple[str, int, int]]):
        """Add chunk tuples (text, start, end) and compute embeddings for them."""
        if not chunks:
            return
        texts = [c[0] for c in chunks]
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if self.embeddings is None:
            self.embeddings = vecs
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack([self.embeddings, vecs])
            self.chunks.extend(chunks)
        # rebuild NN index
        self.nn = NearestNeighbors(n_neighbors=min(TOP_K, len(self.chunks)), metric="cosine")
        self.nn.fit(self.embeddings)

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, int, int, float]]:
        """Return top_k chunks and distances. Each item is (text, start, end, score)
           Score is cosine distance (lower is better). We convert to similarity for readability.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        qvec = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        k = min(top_k, len(self.chunks))
        dists, idxs = self.nn.kneighbors(qvec, n_neighbors=k, return_distance=True)
        dists = dists[0]
        idxs = idxs[0]
        results = []
        for dist, idx in zip(dists, idxs):
            text, start, end = self.chunks[idx]
            sim = 1 - dist  # cosine similarity approx
            results.append((text, start, end, float(sim)))
        return results



def call_gemini(prompt_text: str) -> str:
    """
    Uses the Google genai client. Expects GOOGLE_API_KEY in environment.
    Returns the response text.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        raise EnvironmentError("Set GOOGLE_API_KEY in the environment before running.")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    # The original snippet used model="gemini-2.5-flash". We'll do the same.
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_text
    )
    # Some genai clients return .text, others .content. Use .text per provided snippet.
    return getattr(response, "text", str(response))



def construct_rag_prompt(retrieved: List[Tuple[str, int, int, float]], question: str, max_context_chars: int = 4000) -> str:
    """
    Build a final prompt for Gemini that:
     - Presents the retrieved context with minimal formatting
     - Instructs the model to answer only from the context and to say "I don't know" if not present
     - Requests concise, source-citing answers (with char offsets)
    """
    header = (
        "You are an assistant that answers questions using ONLY the provided context extracted from a PDF.\n"
        "If the answer is not contained in the context, say \"I don't know\". Be concise and accurate.\n"
        "Cite the source chunks as [start:end] character offsets.\n\n"
    )

    ctx_parts = []
    total = 0
    for text, start, end, sim in retrieved:
        # include a small header for each chunk showing offsets and similarity
        block = f"[{start}:{end}] (score={sim:.3f})\n{text.strip()}\n"
        if total + len(block) > max_context_chars:
            break
        ctx_parts.append(block)
        total += len(block)
    context = "\n---\n".join(ctx_parts).strip()
    final_prompt = (
        header +
        "CONTEXT:\n" +
        (context if context else "(no context available)") +
        "\n\nQUESTION:\n" +
        question +
        "\n\nINSTRUCTIONS:\n"
        "  - Answer using only the context above.\n"
        "  - If you cannot find the answer in the context, reply: \"I don't know\".\n"
        "  - Keep the answer short. If multiple facts come from different chunks, cite offsets like [start:end].\n"
    )
    return final_prompt


def main():
    parser = argparse.ArgumentParser(description="Simple RAG with PDF + Google Gemini")
    parser.add_argument("pdf_path", help="Path to PDF file to ingest")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve for RAG")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Overlap between chunks in characters")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    if not os.path.isfile(pdf_path):
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    print("Extracting text from PDF...")
    raw = extract_text_from_pdf(pdf_path)
    raw = clean_text(raw)
    if not raw:
        print("No text extracted from PDF (maybe it's scanned or image-only). Exiting.")
        sys.exit(1)

    print("Chunking document...")
    chunks = chunk_text(raw, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks and building index (this may take a moment)...")
    rag = SimpleRAGIndex(embed_model_name=EMBED_MODEL_NAME)
    rag.add_chunks(chunks)
    print("Index ready.")

    # interactive Q&A loop
    print("\nReady. Ask questions about the PDF. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            raw_q = input("\nYour question> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        q = raw_q.strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        q_clean = clean_question(q)
        # retrieve
        retrieved = rag.retrieve(q_clean, top_k=args.top_k)
        if not retrieved:
            print("No relevant content found in the document.")
            continue
        prompt_text = construct_rag_prompt(retrieved, q_clean)
        # call Gemini
        print("\n[Sending to Gemini — using retrieved PDF context]")
        try:
            answer = call_gemini(prompt_text)
            print("\nGemini answer:\n")
            print(answer)
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            print("If you see a problem here, make sure GOOGLE_API_KEY is set and the genai library is installed.")

if __name__ == "__main__":
    main()
