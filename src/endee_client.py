import os
import json 
import math
import requests
import re
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
ENDEE_BASE_URL   = os.getenv("ENDEE_BASE_URL", "http://localhost:8080")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
EMBEDDING_DIM    = 384

def _headers() -> dict:
    """Only includes Authorization if a token actually exists."""
    h = {"Content-Type": "application/json"}
    # Check if token exists and isn't just whitespace
    if ENDEE_AUTH_TOKEN and ENDEE_AUTH_TOKEN.strip():
        token = ENDEE_AUTH_TOKEN if ENDEE_AUTH_TOKEN.startswith("Bearer ") else f"Bearer {ENDEE_AUTH_TOKEN}"
        h["Authorization"] = token
    return h

def sanitize_index_name(name: str) -> str:
    """Cleans names for Vector DB compatibility (no spaces/caps)."""
    name = os.path.splitext(name)[0]
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    return clean_name

# ─── Index Management (The missing functions are here!) ──────────────────────

def list_indexes() -> list:
    """Returns a list of all existing index names."""
    try:
        r = requests.get(f"{ENDEE_BASE_URL}/api/v1/index/list", headers=_headers(), timeout=10)
        if r.status_code == 200:
            data = r.json()
            indexes = data if isinstance(data, list) else data.get("indexes", data.get("data", []))
            
            names = []
            for idx in indexes:
                if isinstance(idx, dict):
                    names.append(idx.get("name", idx.get("index_name", "")))
                else:
                    names.append(str(idx))
            return names
    except Exception:
        pass
    return []

def index_exists(index_name: str) -> bool:
    """Check if a given index already exists."""
    safe_name = sanitize_index_name(index_name)
    existing_indexes = list_indexes()
    return safe_name in existing_indexes
def create_index(index_name: str, dimension: int = EMBEDDING_DIM, metric: str = "cosine") -> dict:
    """Create an Endee index using the server's current REST schema."""
    url = f"{ENDEE_BASE_URL}/api/v1/index/create"
    safe_name = sanitize_index_name(index_name)

    payload = {
        "index_name": safe_name,
        "dim": int(dimension),
        "space_type": metric.lower(),
        "M": 16,
        "ef_con": 64,
        "precision": "int16",
        "sparse_model": "None",
    }

    try:
        response = requests.post(url, json=payload, headers=_headers(), timeout=15)
        
        if response.status_code in (200, 201):
            try:
                return response.json()
            except ValueError:
                return {"status": "created", "name": safe_name, "message": response.text}
        elif response.status_code == 409:
            return {"status": "exists", "name": safe_name}

        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to create index '{safe_name}': {str(e)}")
def delete_index(index_name: str) -> dict:
    safe_name = sanitize_index_name(index_name)
    url = f"{ENDEE_BASE_URL}/api/v1/index/{safe_name}/delete"
    try:
        response = requests.delete(url, headers=_headers(), timeout=10)
        return {"status": "deleted"} if response.status_code < 300 else {"status": "error"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ─── Vector Operations ────────────────────────────────────────────────────────

def batch_insert_vectors(index_name: str, records: list) -> dict:
    safe_name = sanitize_index_name(index_name)
    url = f"{ENDEE_BASE_URL}/api/v1/index/{safe_name}/vector/insert"
    
    endee_vectors = []
    for record in records:
        vector = record["vector"]
        endee_vectors.append({
            "id": str(record["id"]),
            "vector": vector,
            "meta": json.dumps(record.get("metadata", {})),
            "filter": "{}",
            "norm": math.sqrt(sum(float(v) * float(v) for v in vector)),
        })

    for i in range(0, len(endee_vectors), 500):
        batch = endee_vectors[i:i + 500]
        r = requests.post(url, json=batch, headers=_headers(), timeout=60)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Insert failed: {r.text}")

    return {"status": "inserted", "count": len(records)}

def search_vectors(index_name: str, query_vector: list, top_k: int = 5) -> list:
    safe_name = sanitize_index_name(index_name)
    url = f"{ENDEE_BASE_URL}/api/v1/index/{safe_name}/search"
    payload = {"vector": query_vector, "k": top_k, "ef": 128}

    try:
        response = requests.post(url, json=payload, headers=_headers(), timeout=15)
        if response.status_code == 200:
            return _parse_search_response(response)
    except Exception:
        pass
    return []


def _parse_search_response(response: requests.Response) -> list:
    content_type = response.headers.get("Content-Type", "")
    if "application/msgpack" in content_type:
        try:
            import msgpack
        except ImportError as exc:
            raise RuntimeError("Install msgpack to parse Endee search results: pip install msgpack") from exc

        data = msgpack.unpackb(response.content, raw=False)
        if isinstance(data, dict):
            raw_results = data.get("results", [])
        elif isinstance(data, list) and data and _looks_like_result_row(data[0]):
            raw_results = data
        else:
            raw_results = data[0] if isinstance(data, list) and data else []

        parsed = []
        for result in raw_results:
            if not _looks_like_result_row(result):
                continue
            similarity, vector_id, meta, *_ = result
            metadata = _decode_metadata(meta)
            parsed.append({"id": vector_id, "score": similarity, "metadata": metadata})
        return parsed

    data = response.json()
    results = data if isinstance(data, list) else data.get("results", [])
    return [
        {"id": r.get("id"), "score": r.get("similarity", r.get("score")), "metadata": r.get("meta", {})}
        for r in results
    ]


def _decode_metadata(meta) -> dict:
    if isinstance(meta, (bytes, bytearray)):
        meta = meta.decode("utf-8", errors="replace")
    elif isinstance(meta, list):
        try:
            meta = bytes(meta).decode("utf-8", errors="replace")
        except ValueError:
            return {}

    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str) and meta:
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            return {}
    return {}


def _looks_like_result_row(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 3
        and isinstance(value[0], (float, int))
        and isinstance(value[1], str)
    )

def ping() -> bool:
    try:
        return requests.get(f"{ENDEE_BASE_URL}/api/v1/health", timeout=5).status_code == 200
    except:
        return False 
