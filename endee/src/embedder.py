"""
embedder.py
-----------
Manages the sentence-transformer model for generating text embeddings.
Uses a singleton pattern so the model is only loaded once per session.
"""

import os
from typing import Union
from dotenv import load_dotenv

load_dotenv()

_model = None
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def get_model():
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    """
    Embed a single string into a dense vector.

    Args:
        text: The input string to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings efficiently in a single batch call.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def get_embedding_dim() -> int:
    """Return the dimensionality of the current embedding model."""
    model = get_model()
    return model.get_sentence_embedding_dimension()
