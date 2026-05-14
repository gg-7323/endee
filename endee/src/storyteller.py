"""
storyteller.py
--------------
Story Mode — Explain complex concepts as illustrated cartoon stories
with simple, memorable characters.

Each concept gets:
  - A cast of cartoon characters (each representing a concept/entity)
  - A short narrative story that walks through the concept step by step
  - Scene descriptions for rendering as cartoon panels
  - A "moral" — what the story teaches

This makes abstract concepts (deadlock, Newton's laws, sorting algorithms, etc.)
intuitive and memorable.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv
from src.retriever import retrieve, format_context

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"


# ─── Character Archetypes ─────────────────────────────────────────────────────
# Reusable cast of characters that map well to common CS/science concepts

CHARACTER_PALETTE = {
    "force":        {"name": "Forcy", "emoji": "💪", "color": "#ef4444", "trait": "pushy and energetic"},
    "mass":         {"name": "Blocky", "emoji": "🧱", "color": "#78716c", "trait": "heavy and stubborn"},
    "acceleration": {"name": "Zippy", "emoji": "⚡", "color": "#f59e0b", "trait": "always speeding up"},
    "friction":     {"name": "Grumpy",  "emoji": "😤", "color": "#6b7280", "trait": "always slowing things down"},
    "process":      {"name": "Procy",  "emoji": "🤖", "color": "#3b82f6", "trait": "always busy working"},
    "resource":     {"name": "Ressy",  "emoji": "🍎", "color": "#10b981", "trait": "everyone wants me"},
    "deadlock":     {"name": "Locky",  "emoji": "🔒", "color": "#8b5cf6", "trait": "loves to freeze everything"},
    "data":         {"name": "Datey",  "emoji": "📦", "color": "#06b6d4", "trait": "loves being organized"},
    "algorithm":    {"name": "Algo",   "emoji": "🧠", "color": "#ec4899", "trait": "follows rules strictly"},
    "memory":       {"name": "Memo",   "emoji": "🗄️", "color": "#84cc16", "trait": "remembers everything"},
    "default_a":    {"name": "Alex",   "emoji": "🧑", "color": "#6366f1", "trait": "curious and brave"},
    "default_b":    {"name": "Benji",  "emoji": "🐣", "color": "#f97316", "trait": "small but clever"},
    "default_c":    {"name": "Cara",   "emoji": "🦊", "color": "#14b8a6", "trait": "wise and helpful"},
}


def generate_story(index_name: str, concept: str) -> dict:
    """
    Generate a cartoon story that explains a concept from the notes.

    Args:
        index_name : Endee index for the notes
        concept    : The concept to explain (e.g., "Newton's Second Law", "Deadlock")

    Returns:
        {
          "concept"    : str,
          "title"      : str  — story title,
          "characters" : list[dict] — cast with name, role, emoji, color, trait,
          "scenes"     : list[dict] — each scene has title, narration, dialogue, visual_desc,
          "moral"      : str  — the one-sentence learning takeaway,
          "summary"    : str  — the actual academic explanation after the story
        }
    """
    # Retrieve relevant chunks
    chunks = retrieve(index_name, concept, top_k=4)
    context = format_context(chunks)

    prompt = f"""You are a creative children's science educator. Explain the concept of "{concept}"
as a fun, simple cartoon story with memorable characters.

Here is the actual academic content about this concept from the student's notes:
{context}

Create a story that makes this concept intuitive. The story must be scientifically accurate
but told through simple cartoon characters and relatable situations.

Return ONLY a JSON object with this exact structure:
{{
  "title": "A fun story title (e.g. 'The Day Forcy Pushed Blocky')",
  "characters": [
    {{
      "name": "Character name",
      "emoji": "one emoji representing them",
      "color": "hex color code",
      "represents": "what concept/entity this character represents",
      "personality": "one short personality trait"
    }}
  ],
  "scenes": [
    {{
      "scene_number": 1,
      "title": "Scene title",
      "setting": "Where this scene takes place (vivid, simple description)",
      "narration": "2-3 sentences of narrator text explaining what happens (and what it means scientifically)",
      "dialogue": [
        {{"character": "Name", "line": "what they say"}},
        {{"character": "Name", "line": "response"}}
      ],
      "visual_desc": "Description of what to draw in this panel (simple, cartoon style)",
      "key_concept": "The scientific concept demonstrated in this scene"
    }}
  ],
  "moral": "One sentence — what the story teaches (the actual scientific principle)",
  "fun_fact": "A surprising or memorable fact about this concept",
  "real_world": "One real-world example where this concept appears in daily life"
}}

Rules:
- 3-5 scenes maximum
- 2-4 characters maximum  
- Keep language simple (grade 8 level)
- Each scene must demonstrate a specific aspect of the concept
- Make characters quirky and memorable
- The story should have a clear beginning, middle, end
- Dialogue should be fun but subtly educational

Return ONLY the JSON. No markdown, no explanation."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        story = json.loads(raw)
    except Exception:
        # Minimal fallback
        story = {
            "title": f"The Story of {concept}",
            "characters": [
                {"name": "Alex", "emoji": "🧑", "color": "#6366f1",
                 "represents": concept, "personality": "curious"}
            ],
            "scenes": [
                {
                    "scene_number": 1,
                    "title": "The Beginning",
                    "setting": "A classroom",
                    "narration": f"Today Alex learns about {concept}.",
                    "dialogue": [{"character": "Alex", "line": f"I want to understand {concept}!"}],
                    "visual_desc": "A student at a desk with a textbook",
                    "key_concept": concept,
                }
            ],
            "moral": f"Understanding {concept} helps us explain the world around us.",
            "fun_fact": "Science is everywhere!",
            "real_world": "Look around you — this concept is at work right now.",
        }

    return {
        "concept": concept,
        **story,
        "source_chunks": chunks[:2],
    }


def list_storyable_concepts(index_name: str) -> list[str]:
    """
    Return a list of concepts from the notes that are good candidates for story mode.
    Prefers concrete, visual, or process-based concepts over abstract ones.
    """
    from src.embedder import embed_text
    from src.endee_client import search_vectors

    query_vec = embed_text("process mechanism how it works step by step cause effect example")
    results = search_vectors(index_name, query_vec, top_k=8)
    chunks = [r.get("metadata", {}).get("text", "") for r in results]
    context = "\n\n".join(chunks[:6])

    prompt = f"""From these notes, list 6-8 concepts that would make good cartoon stories.
Prefer: processes, cause-and-effect relationships, mechanisms with steps, laws with examples.
Avoid: pure definitions, abstract math formulas with no physical meaning.

Notes:
{context}

Return ONLY a JSON array of concept name strings.
Example: ["Newton's Second Law", "Friction", "Inertia"]"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        import re
        return re.findall(r'"([^"]+)"', raw)[:8]
