"""
retriever.py
------------
Retrieval layer of the RAG pipeline.
Embeds a user question and fetches the most semantically similar
chunks from Endee.
"""

from src.embedder import embed_text
from src.endee_client import search_vectors


def retrieve(index_name: str, question: str, top_k: int = 5) -> list[dict]:
    """
    Embed the question and retrieve the top_k most relevant chunks from Endee.

    Args:
        index_name : The Endee index to search (corresponds to an uploaded file)
        question   : The user's natural language question
        top_k      : Number of chunks to retrieve

    Returns:
        List of result dicts, each containing:
            - text         : The chunk text
            - score        : Cosine similarity score (higher = more relevant)
            - source_file  : The original file this chunk came from
            - chunk_index  : Position of this chunk in the document
    """
    query_vector = embed_text(question)
    raw_results = search_vectors(index_name, query_vector, top_k=top_k)

    parsed = []
    for result in raw_results:
        metadata = result.get("metadata", {})
        parsed.append({
            "text": metadata.get("text", ""),
            "score": round(result.get("score", 0.0), 4),
            "source_file": metadata.get("source_file", "Unknown"),
            "chunk_index": metadata.get("chunk_index", -1),
            "word_count": metadata.get("word_count", 0),
        })

    # Sort by score descending (Endee may already do this, but ensure it)
    parsed.sort(key=lambda x: x["score"], reverse=True)
    return parsed


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a clean context string for the LLM prompt.
    Each chunk is labeled with its source and chunk number.
    """
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(
            f"[Source {i} | File: {chunk['source_file']} | "
            f"Chunk #{chunk['chunk_index']} | Relevance: {chunk['score']:.2%}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)
