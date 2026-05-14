"""
ingest.py
---------
Handles the full ingestion pipeline:
  1. Parse uploaded PDF or plain-text files
  2. Split into overlapping chunks
  3. Embed each chunk using sentence-transformers
  4. Store vectors + metadata in Endee

This is the R (Retrieval) setup phase of our RAG pipeline.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF

from src.embedder import embed_batch, get_embedding_dim
from src.endee_client import (
    create_index,
    batch_insert_vectors,
    index_exists,
    delete_index,
)

# ─── Chunking Configuration ───────────────────────────────────────────────────

CHUNK_SIZE = 400        # tokens / characters per chunk (approx)
CHUNK_OVERLAP = 80      # overlap between consecutive chunks
INDEX_PREFIX = "notes_" # all note indexes are prefixed for easy identification


# ─── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
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


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks for better retrieval coverage.

    Each chunk carries positional metadata so we can cite the source later.

    Returns:
        List of dicts: { "text": str, "chunk_index": int, "char_start": int, "char_end": int }
    """
    text = clean_text(text)
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        # Calculate approximate character positions for source citation
        char_start = len(" ".join(words[:start]))
        char_end = char_start + len(chunk_text_str)

        chunks.append({
            "text": chunk_text_str,
            "chunk_index": len(chunks),
            "char_start": char_start,
            "char_end": char_end,
            "word_count": len(chunk_words),
        })

        if end == len(words):
            break
        start += chunk_size - overlap  # slide window with overlap

    return chunks


# ─── Index Name Helpers ───────────────────────────────────────────────────────
def make_index_name(filename: str) -> str:
    import re
    from pathlib import Path

    base = Path(filename).name.lower()  # use full name, not stem
    base = base.replace(".", "_")       # remove dots completely

    safe = re.sub(r"[^a-z0-9_]", "_", base)
    safe = re.sub(r"_+", "_", safe).strip("_")

    # Ensure it starts with a letter
    if not safe or not safe[0].isalpha():
        safe = "doc_" + safe

    # Limit length (Endee can be picky)
    safe = safe[:40]

    return f"notes_{safe}"

# ─── Main Ingestion Pipeline ──────────────────────────────────────────────────

def ingest_file(
    file_bytes: bytes,
    filename: str,
    overwrite: bool = False,
    progress_callback=None,
) -> dict:
    """
    Full ingestion pipeline for a single file.

    Steps:
      1. Extract text
      2. Chunk text
      3. Create Endee index
      4. Embed all chunks (batched)
      5. Insert all vectors into Endee

    Args:
        file_bytes       : Raw file content
        filename         : Original filename (used for index naming and metadata)
        overwrite        : If True, delete existing index before re-ingesting
        progress_callback: Optional callable(current, total, message) for UI updates

    Returns:
        Summary dict with index_name, chunk_count, etc.
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

    BATCH_SIZE = 64
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
            "id": f"{index_name}_chunk_{i}_{uuid.uuid4().hex[:8]}",
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

    # Insert in batches of 100
    for i in range(0, len(records), 100):
        batch_insert_vectors(index_name, records[i:i + 100])

    if progress_callback:
        progress_callback(5, 5, "✅ Ingestion complete!")

    return {
        "index_name": index_name,
        "filename": filename,
        "chunk_count": len(chunks),
        "total_words": sum(c["word_count"] for c in chunks),
        "raw_text_preview": raw_text[:500],
    }
