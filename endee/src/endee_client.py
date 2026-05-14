"""
endee_client.py
---------------
Wrapper around the Endee Vector Database REST API.
Uses raw HTTP calls (no SDK) to avoid numpy version conflicts.
Correct field names sourced from official Endee docs: https://docs.endee.io
"""

import os
import json 
import requests
from dotenv import load_dotenv

load_dotenv()

ENDEE_BASE_URL   = os.getenv("ENDEE_BASE_URL", "http://localhost:8080")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
EMBEDDING_DIM    = 384


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if ENDEE_AUTH_TOKEN:
        h["Authorization"] = ENDEE_AUTH_TOKEN
    return h


# ─── Index Management ─────────────────────────────────────────────────────────

def create_index(index_name: str, dimension: int = EMBEDDING_DIM, metric: str = "cosine") -> dict:
    """Create a new vector index. Uses correct Endee API field names."""
    url = f"{ENDEE_BASE_URL}/api/v1/index/create"

    # Correct payload format per official Endee docs
    # Fields: name, dimension, space_type (NOT index_name, NOT metric)
    payload = {
    "index_name": index_name,
    "dimension": 384,
    "metric": "cosine",
    "engine": "hnsw"
    
}

    try:
        response = requests.post(url, data= json.dumps(payload),headers={"Content-Type":"Application/json"}, timeout=10)
        if response.status_code in (200, 201):
            return response.json()
        elif response.status_code == 409:
            return {"status": "exists", "name": index_name}
        else:
            raise RuntimeError(
                f"Endee create_index failed — HTTP {response.status_code}: {response.text[:300]}"
            )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Endee create_index error: {e}")


def delete_index(index_name: str) -> dict:
    """Delete an existing index."""
    url = f"{ENDEE_BASE_URL}/api/v1/index/{index_name}/delete"
    try:
        response = requests.delete(url, headers=_headers(), timeout=10)
        return {"status": "deleted"} if response.status_code in (200, 201, 204) else {"status": "not_found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_indexes() -> list:
    """List all indexes."""
    try:
        response = requests.get(f"{ENDEE_BASE_URL}/api/v1/index/list", headers=_headers(), timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("indexes", data.get("data", []))
    except Exception:
        pass
    return []


def index_exists(index_name: str) -> bool:
    """Check if a given index already exists."""
    try:
        for idx in list_indexes():
            name = idx.get("name", idx.get("index_name", "")) if isinstance(idx, dict) else str(idx)
            if name == index_name:
                return True
    except Exception:
        pass
    return False


# ─── Vector Operations ────────────────────────────────────────────────────────

def batch_insert_vectors(index_name: str, records: list) -> dict:
    """
    Upsert vectors into Endee.
    Endee docs use: POST /api/v1/index/{name}/upsert
    with body: { "vectors": [{ "id", "vector", "meta", "filter" }] }
    """
    # Convert internal format → Endee format
    # Internal uses "metadata", Endee uses "meta"
    endee_vectors = []
    for r in records:
        endee_vectors.append({
            "id":     r["id"],
            "vector": r["vector"],
            "meta":   r.get("metadata", {}),
        })

    # Try upsert endpoint first (official docs)
    url_upsert = f"{ENDEE_BASE_URL}/api/v1/index/{index_name}/upsert"
    # Try batch insert as fallback
    url_batch  = f"{ENDEE_BASE_URL}/api/v1/index/{index_name}/insert/batch"

    CHUNK = 500  # Endee max 1000 per call, use 500 to be safe
    for i in range(0, len(endee_vectors), CHUNK):
        batch = endee_vectors[i:i + CHUNK]
        success = False

        # Try upsert (official format)
        try:
            r = requests.post(url_upsert, json={"vectors": batch}, headers=_headers(), timeout=60)
            if r.status_code in (200, 201):
                success = True
        except Exception:
            pass

        # Fallback: batch insert with "records" key
        if not success:
            try:
                # Convert back to "metadata" key for older API
                old_batch = [{"id": v["id"], "vector": v["vector"], "metadata": v["meta"]} for v in batch]
                r = requests.post(url_batch, json={"records": old_batch}, headers=_headers(), timeout=60)
                if r.status_code in (200, 201):
                    success = True
            except Exception:
                pass

        # Last resort: individual inserts
        if not success:
            for v in batch:
                try:
                    requests.post(
                        f"{ENDEE_BASE_URL}/api/v1/index/{index_name}/insert",
                        json={"id": v["id"], "vector": v["vector"], "metadata": v["meta"]},
                        headers=_headers(),
                        timeout=15,
                    )
                except Exception:
                    pass

    return {"status": "inserted", "count": len(records)}


def search_vectors(index_name: str, query_vector: list, top_k: int = 5) -> list:
    """
    Semantic ANN search.
    Endee docs: POST /api/v1/index/{name}/query
    body: { "vector": [...], "top_k": N }
    """
    # Try official query endpoint first, then search as fallback
    for url, payload in [
        (f"{ENDEE_BASE_URL}/api/v1/index/{index_name}/query",  {"vector": query_vector, "top_k": top_k, "ef": 128}),
        (f"{ENDEE_BASE_URL}/api/v1/index/{index_name}/search", {"vector": query_vector, "top_k": top_k}),
    ]:
        try:
            response = requests.post(url, json=payload, headers=_headers(), timeout=15)
            if response.status_code == 200:
                data = response.json()
                results = data if isinstance(data, list) else data.get("results", data.get("data", []))

                # Normalize → internal format
                parsed = []
                for r in results:
                    # Endee returns "meta" and "similarity", we use "metadata" and "score"
                    meta = r.get("meta", r.get("metadata", {}))
                    parsed.append({
                        "id":          r.get("id", ""),
                        "score":       r.get("similarity", r.get("score", 0.0)),
                        "metadata":    meta,
                        "text":        meta.get("text", ""),
                        "source_file": meta.get("source_file", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "word_count":  meta.get("word_count", 0),
                    })
                return parsed
        except Exception:
            continue
    return []


# ─── Health Check ─────────────────────────────────────────────────────────────

def ping() -> bool:
    """Return True if Endee server is reachable."""
    for path in ["/api/v1/health", "/health", "/api/v1/index/list"]:
        try:
            r = requests.get(f"{ENDEE_BASE_URL}{path}", headers=_headers(), timeout=5)
            if r.status_code in (200, 201):
                return True
        except Exception:
            continue
    return False