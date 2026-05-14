"""
ingest.py
---------
Handles the full ingestion pipeline:
  1. Parse uploaded PDF or plain-text files
  2. Split into overlapping chunks
  3. Embed each chunk using sentence-transformers
  4. Store vectors + metadata in Endee
"""

import os
import re
import uuid
from pathlib import Path

# PyMuPDF import — handles both old (fitz) and new (pymupdf) package names
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

from src.embedder import embed_batch, get_embedding_dim
from src.endee_client import (
    create_index,
    batch_insert_vectors,
    index_exists,
    delete_index,
)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
INDEX_PREFIX = "notes_"


# ─── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    if fitz is None:
        raise ImportError("PyMuPDF is not installed. Run: pip install PyMuPDF")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            full_text.append(f"[Page {page_num + 1}]\n{text.strip()}")
    return "\n\n".join(full_text)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode plain text files."""
    return file_bytes.decode("utf-8", errors="replace")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Route to the correct extractor based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF, TXT, or MD.")


# ─── Chunking ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize line endings."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks for better retrieval coverage."""
    text = clean_text(text)
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text_str,
            "chunk_index": len(chunks),
            "word_count": len(chunk_words),
        })

        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


# ─── Index Name Helper ────────────────────────────────────────────────────────

def make_index_name(filename: str) -> str:
    """Create a safe, deterministic Endee index name from a filename."""
    base = Path(filename).stem.lower()
    safe = re.sub(r"[^a-z0-9_]", "_", base)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return f"{INDEX_PREFIX}{safe}"[:50]


# ─── Main Ingestion Pipeline ──────────────────────────────────────────────────

def ingest_file(
    file_bytes: bytes,
    filename: str,
    overwrite: bool = False,
    progress_callback=None,
) -> dict:
    """
    Full ingestion pipeline for a single file.
    Steps: Extract → Chunk → Create Index → Embed → Insert into Endee
    """
    index_name = make_index_name(filename)

    # Step 1: Extract text
    if progress_callback:
        progress_callback(0, 5, "📄 Extracting text from file...")
    raw_text = extract_text(file_bytes, filename)

    if not raw_text.strip():
        raise ValueError("No text could be extracted from this file.")

    # Step 2: Chunk text
    if progress_callback:
        progress_callback(1, 5, "✂️ Splitting into chunks...")
    chunks = chunk_text(raw_text)

    if not chunks:
        raise ValueError("File too short to process.")

    # Step 3: Create Endee index
    if progress_callback:
        progress_callback(2, 5, "🗄️ Setting up Endee vector index...")

    if overwrite and index_exists(index_name):
        delete_index(index_name)

    dim = get_embedding_dim()
    create_index(index_name, dimension=dim, metric="cosine")

    # Step 4: Embed all chunks in batches
    if progress_callback:
        progress_callback(3, 5, f"🧠 Embedding {len(chunks)} chunks...")

    BATCH_SIZE = 32
    all_embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_texts = [c["text"] for c in chunks[i:i + BATCH_SIZE]]
        batch_embeddings = embed_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)

    # Step 5: Insert into Endee
    if progress_callback:
        progress_callback(4, 5, "📤 Storing vectors in Endee...")

    records = []
    for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        record = {
            "id": f"{index_name}_{i}_{uuid.uuid4().hex[:6]}",
            "vector": embedding,
            "metadata": {
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"],
                "word_count": chunk["word_count"],
                "source_file": filename,
                "index_name": index_name,
            },
        }
        records.append(record)

    # Insert in batches of 50
    for i in range(0, len(records), 50):
        batch_insert_vectors(index_name, records[i:i + 50])

    if progress_callback:
        progress_callback(5, 5, "✅ Ingestion complete!")

    return {
        "index_name": index_name,
        "filename": filename,
        "chunk_count": len(chunks),
        "total_words": sum(c["word_count"] for c in chunks),
        "raw_text_preview": raw_text[:500],
    }