"""
rag.py
------
Generation layer of the RAG pipeline.
Takes retrieved context chunks and uses Groq (llama-3.3-70b-versatile)
to generate a grounded, accurate answer.

The LLM is strictly instructed to answer ONLY from the provided context —
it will not hallucinate beyond the notes.
"""

import os
from groq import Groq
from dotenv import load_dotenv

from src.retriever import retrieve, format_context

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an intelligent study assistant. Your job is to answer students' 
questions based STRICTLY on the provided lecture notes and study material.

RULES you must follow:
1. Answer ONLY using information from the provided context (the notes).
2. If the answer is NOT found in the notes, say exactly: 
   "⚠️ This topic doesn't appear to be covered in your uploaded notes. 
    Try asking something related to the content in your file."
3. Be clear, concise, and educational in tone.
4. If relevant, mention which source/chunk the answer came from.
5. Use bullet points or numbered lists when explaining multi-step concepts.
6. Never make up facts, definitions, or formulas not present in the notes.
7. If the context partially answers the question, share what you found and 
   indicate that the notes may not have full coverage of the topic.

Your answers should feel like a knowledgeable tutor explaining from the student's own material."""


def answer_question(
    index_name: str,
    question: str,
    top_k: int = 5,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Full RAG pipeline: retrieve → augment → generate.

    Args:
        index_name   : Endee index to retrieve from
        question     : Student's question
        top_k        : Number of chunks to retrieve
        chat_history : Optional list of {"role": ..., "content": ...} for multi-turn

    Returns:
        {
          "answer"           : str   — LLM-generated answer
          "retrieved_chunks" : list  — raw chunks used as context
          "context_used"     : str   — formatted context string sent to LLM
          "question"         : str   — the original question
        }
    """

    # ── Step 1: Retrieve relevant chunks from Endee ──────────────────────────
    retrieved_chunks = retrieve(index_name, question, top_k=top_k)

    if not retrieved_chunks:
        return {
            "answer": "⚠️ No relevant content found in the notes. Please check that the file was ingested correctly.",
            "retrieved_chunks": [],
            "context_used": "",
            "question": question,
        }

    # ── Step 2: Format context for prompt ────────────────────────────────────
    context = format_context(retrieved_chunks)

    # ── Step 3: Build messages for Claude ────────────────────────────────────
    user_message = f"""Here are the relevant sections from the student's notes:

<context>
{context}
</context>

Student's Question: {question}

Please answer based only on the context above."""

    messages = []

    # Include chat history for multi-turn conversations
    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_message})

    # ── Step 4: Generate answer with Groq ────────────────────────────────────
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1024,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "context_used": context,
        "question": question,
    }


def generate_quiz(index_name: str, num_questions: int = 5) -> str:
    """
    Bonus feature: auto-generate a quiz from the uploaded notes.
    Retrieves a broad sample of content and asks Claude to create MCQs.
    """
    from src.embedder import embed_text
    from src.endee_client import search_vectors

    # Use a generic query to get a broad spread of content
    sample_query = embed_text("key concepts definitions formulas important points")
    results = search_vectors(index_name, sample_query, top_k=10)

    chunks = [r.get("metadata", {}).get("text", "") for r in results]
    context = "\n\n---\n\n".join(chunks[:8])

    prompt = f"""Based on these study notes, generate {num_questions} multiple choice questions 
to help a student test their understanding.

Notes:
{context}

Format each question as:
Q1. [Question]
A) [Option]
B) [Option]  
C) [Option]
D) [Option]
Answer: [Correct letter] — [Brief explanation]

Make the questions test genuine understanding, not just memorization."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def summarize_notes(index_name: str) -> str:
    """
    Bonus feature: generate a structured summary of the uploaded notes.
    """
    from src.embedder import embed_text
    from src.endee_client import search_vectors

    sample_query = embed_text("introduction overview main topics summary")
    results = search_vectors(index_name, sample_query, top_k=12)

    chunks = [r.get("metadata", {}).get("text", "") for r in results]
    context = "\n\n".join(chunks[:10])

    prompt = f"""Read these lecture notes and produce a structured summary.

Notes:
{context}

Provide:
1. **Topic**: What subject/chapter is this about?
2. **Key Concepts** (bullet list of 5-8 main ideas)
3. **Important Definitions** (any terms defined in the notes)
4. **Key Formulas or Rules** (if any)
5. **Quick Recap** (2-3 sentence summary a student can read before an exam)"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content
