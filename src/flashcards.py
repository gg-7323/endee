"""
flashcards.py
-------------
Flashcard & Mind Map Generator

Features:
  1. Flashcard Generator — creates Q&A cards from notes for spaced repetition study
  2. Mind Map Generator — builds a hierarchical topic map from the uploaded notes

Both are grounded in the notes via Endee retrieval.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

from src.embedder import embed_text
from src.endee_client import search_vectors

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"


# ─── Flashcard Generator ──────────────────────────────────────────────────────

def generate_flashcards(index_name: str, num_cards: int = 10, difficulty: str = "mixed") -> list[dict]:
    """
    Generate flashcards from the uploaded notes.

    Each flashcard has:
        - front: A question or prompt
        - back: The answer (from notes)
        - category: Topic category
        - difficulty: easy | medium | hard
        - hint: A small hint to guide thinking

    Args:
        index_name : Endee index for the notes
        num_cards  : Number of cards to generate
        difficulty : "easy" | "medium" | "hard" | "mixed"

    Returns:
        List of flashcard dicts
    """
    query_vec = embed_text("definitions formulas concepts laws rules examples key facts")
    results = search_vectors(index_name, query_vec, top_k=12)
    chunks = [r.get("metadata", {}).get("text", "") for r in results]
    context = "\n\n---\n\n".join(chunks[:10])

    difficulty_guide = {
        "easy": "simple recall questions — definitions and basic facts",
        "medium": "application questions — applying concepts to examples",
        "hard": "analysis questions — comparing, contrasting, explaining why",
        "mixed": "a mix of easy (40%), medium (40%), and hard (20%) questions",
    }.get(difficulty, "a mix of difficulty levels")

    prompt = f"""You are a study card creator. Create {num_cards} flashcards from these study notes.
Difficulty level: {difficulty_guide}

Notes:
{context}

Return ONLY a JSON array of flashcard objects. Each object must have:
{{
  "front": "The question or prompt on the front of the card",
  "back": "The complete answer on the back of the card",
  "category": "The topic/subtopic this card belongs to",
  "difficulty": "easy | medium | hard",
  "hint": "A small hint that helps recall without giving away the answer"
}}

Rules:
- Questions must be answerable from the notes only
- Mix definition cards, formula cards, and application cards
- Keep answers concise but complete
- Use clear, exam-style language

Return ONLY the JSON array. No markdown fences, no explanation."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        cards = json.loads(raw)
        return cards[:num_cards]
    except Exception:
        # Fallback parsing
        import re
        objects = re.findall(r'\{[^{}]+\}', raw, re.DOTALL)
        cards = []
        for obj in objects:
            try:
                cards.append(json.loads(obj))
            except Exception:
                continue
        return cards[:num_cards]


# ─── Mind Map Generator ───────────────────────────────────────────────────────

def generate_mindmap(index_name: str) -> dict:
    """
    Generate a hierarchical mind map from the uploaded notes.

    Returns a nested dict structure:
    {
      "root": "Main Topic",
      "children": [
        {
          "topic": "Subtopic 1",
          "color": "#7c6fff",
          "children": [
            {"topic": "Concept A", "detail": "brief detail", "children": []},
            ...
          ]
        },
        ...
      ]
    }
    """
    query_vec = embed_text("introduction overview main topics structure chapters sections")
    results = search_vectors(index_name, query_vec, top_k=12)
    chunks = [r.get("metadata", {}).get("text", "") for r in results]
    context = "\n\n".join(chunks[:10])

    prompt = f"""Analyze these study notes and create a hierarchical mind map structure.

Notes:
{context}

Return ONLY a JSON object representing the mind map. Structure:
{{
  "root": "The main subject/title of these notes",
  "children": [
    {{
      "topic": "Major topic or section",
      "color": "one of: #7c6fff | #4ade80 | #f59e0b | #60a5fa | #f472b6 | #34d399",
      "children": [
        {{
          "topic": "Subtopic or concept",
          "detail": "one-line description or key fact",
          "children": [
            {{"topic": "specific detail or formula", "detail": "", "children": []}}
          ]
        }}
      ]
    }}
  ]
}}

Rules:
- Root = the overall subject of the notes
- 3-5 major branches (main sections/topics)
- Each branch has 2-4 sub-nodes
- Each sub-node may have 1-3 leaf nodes (formulas, examples, definitions)
- Keep topic names SHORT (3-5 words max)
- Detail field: 1 sentence max or empty string
- Assign different colors to each major branch

Return ONLY the JSON. No markdown, no explanation."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        mindmap = json.loads(raw)
        return mindmap
    except Exception:
        return {
            "root": "Study Notes",
            "children": [
                {
                    "topic": "Key Concepts",
                    "color": "#7c6fff",
                    "children": [{"topic": "See notes for details", "detail": "", "children": []}]
                }
            ]
        }
