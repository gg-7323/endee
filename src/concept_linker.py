"""
concept_linker.py
-----------------
Concept Linking — discovers and explains semantic relationships
between topics mentioned in the user's notes.

How it works:
  1. Extract key concepts from the notes using Claude
  2. For a given query like "How is deadlock related to resource allocation?",
     retrieve relevant chunks for EACH concept separately from Endee
  3. Feed both sets of chunks to Claude and ask it to reason about the connection
  4. Return a structured relationship graph + natural language explanation

This goes beyond simple retrieval — it adds a semantic reasoning layer
that shows how ideas in the notes connect to each other.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

from src.retriever import retrieve, format_context
from src.embedder import embed_text
from src.endee_client import search_vectors

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"


# ─── Concept Extraction ───────────────────────────────────────────────────────

def extract_concepts(index_name: str, max_concepts: int = 12) -> list[str]:
    """
    Extract the main concepts/topics from the uploaded notes.
    Uses a broad semantic search + Claude to identify key terms.

    Returns:
        List of concept strings (e.g. ["Newton's First Law", "Friction", "Momentum"])
    """
    # Sample notes broadly
    query_vec = embed_text("key concepts topics definitions important terms main ideas")
    results = search_vectors(index_name, query_vec, top_k=10)
    chunks = [r.get("metadata", {}).get("text", "") for r in results]
    context = "\n\n".join(chunks[:8])

    prompt = f"""From the following study notes, extract the {max_concepts} most important 
concepts, topics, or terms that a student should understand.

Notes:
{context}

Return ONLY a JSON array of strings. Example: ["Newton's First Law", "Momentum", "Friction"]
No explanation, no markdown — just the raw JSON array."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()
    # Strip any accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        concepts = json.loads(raw)
        return [str(c) for c in concepts[:max_concepts]]
    except Exception:
        # Fallback: extract quoted strings
        import re
        return re.findall(r'"([^"]+)"', raw)[:max_concepts]


# ─── Relationship Detection ───────────────────────────────────────────────────

def find_relationship(
    index_name: str,
    concept_a: str,
    concept_b: str,
) -> dict:
    """
    Find and explain the semantic relationship between two concepts
    as described in the user's notes.

    Returns:
        {
          "concept_a"      : str,
          "concept_b"      : str,
          "relationship"   : str,  — one of: "causes", "is_part_of", "contrasts_with",
                                              "depends_on", "leads_to", "defines", "related_to"
          "explanation"    : str,  — natural language explanation
          "strength"       : float — 0.0 to 1.0, how strongly related
          "chunks_used"    : list  — source chunks
        }
    """
    # Retrieve relevant chunks for each concept
    chunks_a = retrieve(index_name, concept_a, top_k=3)
    chunks_b = retrieve(index_name, concept_b, top_k=3)

    context_a = format_context(chunks_a)
    context_b = format_context(chunks_b)

    prompt = f"""You are analyzing study notes to find the relationship between two concepts.

Concept A: "{concept_a}"
Relevant notes about A:
{context_a}

---

Concept B: "{concept_b}"
Relevant notes about B:
{context_b}

---

Based ONLY on the notes above, determine how these two concepts are related.

Return a JSON object with this exact structure:
{{
  "relationship": "<one of: causes | is_part_of | contrasts_with | depends_on | leads_to | defines | enables | related_to>",
  "explanation": "<2-3 sentence explanation of how A and B are related, grounded in the notes>",
  "strength": <float between 0.0 and 1.0 — how strongly related>,
  "direction": "<A_to_B | B_to_A | bidirectional>",
  "key_link": "<the single most important connecting idea in one short phrase>"
}}

Return ONLY the JSON. No markdown, no explanation."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
    except Exception:
        result = {
            "relationship": "related_to",
            "explanation": "These concepts appear in the same notes and are related.",
            "strength": 0.5,
            "direction": "bidirectional",
            "key_link": "shared context",
        }

    return {
        "concept_a": concept_a,
        "concept_b": concept_b,
        **result,
        "chunks_used": chunks_a[:2] + chunks_b[:2],
    }


def answer_concept_link_question(index_name: str, question: str) -> dict:
    """
    Answer a relationship question like "How is X related to Y?" or
    "What is the connection between A and B?".

    Automatically parses the concepts from the question and
    performs multi-concept retrieval + reasoning.

    Returns:
        {
          "concepts"     : [str, str],
          "relationship" : dict from find_relationship(),
          "answer"       : str — full natural language answer,
          "graph_data"   : dict — data for visual graph rendering
        }
    """
    # Step 1: Extract the two concepts from the question
    parse_prompt = f"""Extract the two main concepts being compared or linked in this question.
Question: "{question}"
Return ONLY a JSON array of exactly 2 strings. Example: ["deadlock", "resource allocation"]
No explanation."""

    parse_resp = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": parse_prompt}],
    )
    raw = parse_resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()

    try:
        import json, re
        concepts = json.loads(raw)
        if len(concepts) < 2:
            raise ValueError
    except Exception:
        import re
        concepts = re.findall(r'"([^"]+)"', raw)
        if len(concepts) < 2:
            concepts = question.replace("?","").split(" and ")[-2:]

    concept_a, concept_b = concepts[0], concepts[1]

    # Step 2: Find relationship
    rel = find_relationship(index_name, concept_a, concept_b)

    # Step 3: Generate full answer
    chunks_a = retrieve(index_name, concept_a, top_k=3)
    chunks_b = retrieve(index_name, concept_b, top_k=3)
    all_chunks = chunks_a + chunks_b
    context = format_context(all_chunks)

    answer_prompt = f"""Based on these study notes, answer this question about how two concepts relate:

Question: {question}

Context from notes:
{context}

Provide a clear, educational explanation that:
1. Defines each concept briefly (from the notes)
2. Explains the specific relationship between them
3. Gives a concrete example or analogy from the notes if available
4. States why understanding this connection matters

Answer only from the notes. Be concise but complete."""

    answer_resp = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": answer_prompt}],
    )

    # Step 4: Build graph data for visualization
    graph_data = {
        "nodes": [
            {"id": concept_a, "type": "concept_a"},
            {"id": concept_b, "type": "concept_b"},
        ],
        "edges": [
            {
                "from": concept_a if rel.get("direction") in ("A_to_B", "bidirectional") else concept_b,
                "to": concept_b if rel.get("direction") in ("A_to_B", "bidirectional") else concept_a,
                "label": rel.get("relationship", "related_to").replace("_", " "),
                "strength": rel.get("strength", 0.5),
                "key_link": rel.get("key_link", ""),
            }
        ],
    }

    return {
        "concepts": [concept_a, concept_b],
        "relationship": rel,
        "answer": answer_resp.choices[0].message.content,
        "graph_data": graph_data,
        "chunks_used": all_chunks,
    }


def build_concept_map(index_name: str, concepts: list[str]) -> dict:
    """
    Build a full concept map showing relationships between all provided concepts.
    Returns node + edge data suitable for rendering a graph.
    """
    nodes = [{"id": c, "type": "concept"} for c in concepts]
    edges = []

    # Only check pairs that are likely related (limit API calls)
    import itertools
    pairs = list(itertools.combinations(concepts[:8], 2))

    for a, b in pairs[:10]:  # max 10 edges
        try:
            rel = find_relationship(index_name, a, b)
            if rel.get("strength", 0) > 0.4:  # only show meaningful connections
                edges.append({
                    "from": a,
                    "to": b,
                    "label": rel.get("relationship", "related_to").replace("_", " "),
                    "strength": rel.get("strength", 0.5),
                    "key_link": rel.get("key_link", ""),
                })
        except Exception:
            continue

    return {"nodes": nodes, "edges": edges}
