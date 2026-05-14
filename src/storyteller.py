"""
storyteller.py
--------------
Story Mode helpers.

Turns retrieved note chunks into a short, memorable cartoon story for a concept.
"""

import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

from src.embedder import embed_text
from src.endee_client import search_vectors
from src.retriever import format_context, retrieve

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"


def _json_from_model(raw: str) -> dict:
    """Extract a JSON object from an LLM response."""
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _fallback_story(concept: str, context: str) -> dict:
    snippet = context[:500].strip()
    return {
        "title": f"{concept.title()} Story",
        "concept": concept,
        "characters": [
            {
                "name": "Curious Student",
                "emoji": "🧑‍🎓",
                "represents": "the learner",
                "personality": "asks clear questions",
                "color": "#7c6fff",
            }
        ],
        "scenes": [
            {
                "scene_number": 1,
                "title": "Finding the Key Idea",
                "setting": "A study desk covered with notes",
                "narration": snippet,
                "dialogue": [
                    {
                        "character": "Curious Student",
                        "line": "Let me connect this idea step by step from the notes.",
                    }
                ],
                "key_concept": concept,
            }
        ],
        "moral": "Break the concept into smaller note-backed ideas and connect them in order.",
        "fun_fact": "",
        "real_world": "",
    }


def _empty_context_error(concept: str) -> dict:
    return {
        "title": f"No note context found for {concept}",
        "concept": concept,
        "characters": [],
        "scenes": [],
        "moral": "",
        "fun_fact": "",
        "real_world": "",
        "error": (
            "Story Mode could not retrieve note chunks for this concept from the active index. "
            "Make sure the correct PDF is selected under Active Notes, then re-ingest it with "
            "'Re-ingest if exists' checked."
        ),
    }


def _chunks_from_search(index_name: str, query: str, top_k: int = 8) -> list[dict]:
    query_vec = embed_text(query)
    results = search_vectors(index_name, query_vec, top_k=top_k)
    chunks = []
    for result in results:
        metadata = result.get("metadata", {})
        text = metadata.get("text", "").strip()
        if not text:
            continue
        chunks.append(
            {
                "text": text,
                "score": round(result.get("score", 0.0), 4),
                "source_file": metadata.get("source_file", "Unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "word_count": metadata.get("word_count", 0),
            }
        )
    return chunks


def _retrieve_story_chunks(index_name: str, concept: str) -> list[dict]:
    queries = [
        concept,
        f"{concept} ethics artificial intelligence human impact",
        "ethics of artificial intelligence human psychology social impact mental health",
        "artificial intelligence effects on humans society privacy bias responsibility",
        "main topics definitions examples important points artificial intelligence ethics",
    ]

    seen = set()
    combined = []
    for i, query in enumerate(queries):
        chunks = retrieve(index_name, query, top_k=8) if i == 0 else _chunks_from_search(index_name, query)
        for chunk in chunks:
            key = (chunk.get("source_file"), chunk.get("chunk_index"), chunk.get("text")[:80])
            if key in seen:
                continue
            seen.add(key)
            combined.append(chunk)
        if len(combined) >= 6:
            break

    combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return combined[:8]


def generate_story(index_name: str, concept: str) -> dict:
    """Generate a cartoon story grounded in retrieved note chunks."""
    concept = concept.strip()
    retrieved_chunks = _retrieve_story_chunks(index_name, concept)
    context = format_context(retrieved_chunks)

    if not context.strip():
        return _empty_context_error(concept)

    prompt = f"""Create a short illustrated cartoon story that explains this concept from the student's notes.

Concept: {concept}

Relevant note excerpts:
{context}

Return ONLY valid JSON with this exact structure:
{{
  "title": "Short story title",
  "concept": "{concept}",
  "characters": [
    {{
      "name": "Character name",
      "emoji": "single emoji",
      "represents": "what concept/entity they represent",
      "personality": "short trait",
      "color": "#7c6fff"
    }}
  ],
  "scenes": [
    {{
      "scene_number": 1,
      "title": "Scene title",
      "setting": "Cartoon panel setting",
      "narration": "2-4 sentences explaining the concept using the notes",
      "dialogue": [
        {{"character": "Character name", "line": "Short line"}}
      ],
      "key_concept": "One note-backed takeaway"
    }}
  ],
  "moral": "The main lesson",
  "fun_fact": "Optional note-backed memory hook",
  "real_world": "Optional real-world connection grounded in the notes"
}}

Rules:
- Use only the note excerpts.
- Make 3-4 scenes.
- Keep language simple and exam-friendly.
- Do not include markdown fences or commentary."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=2200,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content
    try:
        story = _json_from_model(raw)
    except Exception:
        story = _fallback_story(concept, context)

    story.setdefault("title", f"{concept.title()} Story")
    story.setdefault("concept", concept)
    story.setdefault("characters", [])
    story.setdefault("scenes", [])
    story.setdefault("moral", "")
    return story


def list_storyable_concepts(index_name: str) -> list[str]:
    """Suggest concepts from the uploaded notes that would work well as stories."""
    query_vec = embed_text("main concepts laws definitions processes examples")
    results = search_vectors(index_name, query_vec, top_k=10)
    chunks = [r.get("metadata", {}).get("text", "") for r in results if r.get("metadata", {}).get("text")]
    context = "\n\n---\n\n".join(chunks[:8])

    if not context.strip():
        return []

    prompt = f"""Read these note excerpts and suggest 6 concepts that would be good for Story Mode.

Notes:
{context}

Return ONLY a JSON array of short concept strings. No markdown."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        concepts = json.loads(raw)
    except Exception:
        concepts = re.findall(r'"([^"]+)"', raw)

    return [str(item).strip() for item in concepts if str(item).strip()][:6]
