# 📚 Notes Q&A Bot — RAG-Powered Study Assistant

> **Upload your lecture notes. Ask anything. Get answers grounded strictly in your material.**  
> Built with [Endee](https://github.com/endee-io/endee) Vector Database + Claude AI + Streamlit.

---

## 🎯 What This Project Does

Notes Q&A Bot is a **Retrieval-Augmented Generation (RAG)** application that transforms static lecture notes (PDF or text) into an interactive AI tutor.

Unlike generic chatbots that rely on pre-trained knowledge, this system **only answers from your uploaded notes** — preventing hallucination and keeping answers grounded in your actual study material.

**Key capabilities:**
- Upload any PDF or `.txt` lecture notes
- Ask natural language questions and get contextual answers
- See *exactly which part of your notes* the answer came from
- Auto-generate a **quiz** from your notes
- Generate a **structured summary** of key concepts, definitions, and formulas
- Multi-turn conversation with memory of previous questions

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                         │
│                                                                    │
│  PDF/TXT File                                                      │
│      │                                                             │
│      ▼                                                             │
│  Text Extraction (PyMuPDF)                                         │
│      │                                                             │
│      ▼                                                             │
│  Chunking (400 words, 80-word overlap)                             │
│      │                                                             │
│      ▼                                                             │
│  Embedding (sentence-transformers: all-MiniLM-L6-v2, dim=384)     │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────────────────────┐                                  │
│  │   ENDEE VECTOR DATABASE     │  ← Stores 384-dim embeddings     │
│  │   (Docker, port 8080)       │     + chunk metadata             │
│  └─────────────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        RAG QUERY PIPELINE                         │
│                                                                    │
│  User Question                                                     │
│      │                                                             │
│      ▼                                                             │
│  Embed Question (same model)                                       │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────────────────────┐                                  │
│  │   ENDEE: ANN Search         │  ← Cosine similarity search      │
│  │   POST /api/v1/index/{}/search │  Returns top-K chunks        │
│  └─────────────────────────────┘                                  │
│      │                                                             │
│      ▼                                                             │
│  Build Prompt with retrieved context                               │
│      │                                                             │
│      ▼                                                             │
│  Claude API (claude-sonnet-4)                                      │
│  "Answer ONLY from the provided notes"                             │
│      │                                                             │
│      ▼                                                             │
│  Grounded Answer + Source Citations                                │
│      │                                                             │
│      ▼                                                             │
│  Streamlit UI                                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ How Endee is Used

Endee is the core vector storage and retrieval engine powering this project. Here's exactly how:

### 1. Index Creation
```python
# One index per uploaded file
requests.post("http://localhost:8080/api/v1/index/create", json={
    "index_name": "notes_physics_newtons_laws",
    "dimension": 384,        # matches all-MiniLM-L6-v2
    "metric": "cosine"       # cosine similarity for semantic search
})
```

### 2. Batch Vector Insertion
```python
# Each chunk of the document becomes a vector
requests.post("http://localhost:8080/api/v1/index/notes_physics/insert/batch", json={
    "records": [
        {
            "id": "notes_physics_chunk_0_ab3f1c",
            "vector": [0.023, -0.154, ...],  # 384-dim embedding
            "metadata": {
                "text": "Newton's First Law states that an object at rest...",
                "chunk_index": 0,
                "source_file": "physics_notes.pdf",
                "word_count": 387
            }
        },
        # ... more chunks
    ]
})
```

### 3. Semantic Search (the RAG "Retrieval" step)
```python
# Embed the user's question and search Endee
query_vector = embed("What is Newton's second law?")   # 384-dim

results = requests.post("http://localhost:8080/api/v1/index/notes_physics/search", json={
    "vector": query_vector,
    "top_k": 5     # return 5 most semantically similar chunks
})
# Returns chunks ranked by cosine similarity score
```

The retrieved chunks (with their cosine scores) are then injected into the Claude prompt as context — this is the "Augmented" part of RAG.

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Anthropic API Key ([get one here](https://console.anthropic.com/))

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/notes-qa-bot-endee.git
cd notes-qa-bot-endee
```

### Step 2: Start Endee Vector Database
```bash
docker compose up -d
# Endee will be running at http://localhost:8080
```

Verify it's running:
```bash
curl http://localhost:8080/api/v1/health
```

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

```env
ANTHROPIC_API_KEY=sk-ant-...
ENDEE_BASE_URL=http://localhost:8080
ENDEE_AUTH_TOKEN=        # leave empty for no auth
```

### Step 5: Run the App
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Step 6: Try It with Sample Notes
The repo includes two sample notes files in `data/sample_notes/`:
- `physics_newtons_laws.txt` — Newton's Laws of Motion (Class 11 Physics)
- `cs_binary_search_trees.txt` — Binary Search Trees (CS Data Structures)

Use the **"Load Sample"** button in the sidebar to try immediately.

---

## 📁 Project Structure

```
notes-qa-bot/
├── app.py                          # Streamlit UI (main entry point)
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Endee vector database
├── .env.example                    # Environment variable template
│
├── src/
│   ├── __init__.py
│   ├── endee_client.py             # Endee REST API wrapper
│   │                               #  - create_index(), search_vectors()
│   │                               #  - batch_insert_vectors(), ping()
│   ├── embedder.py                 # sentence-transformers wrapper
│   │                               #  - embed_text(), embed_batch()
│   ├── ingest.py                   # Ingestion pipeline
│   │                               #  - PDF/TXT extraction (PyMuPDF)
│   │                               #  - Chunking with overlap
│   │                               #  - Embedding + storing in Endee
│   ├── retriever.py                # Semantic search layer
│   │                               #  - retrieve() → Endee ANN search
│   │                               #  - format_context() for LLM prompt
│   └── rag.py                      # RAG generation layer
│                                   #  - answer_question() — full pipeline
│                                   #  - generate_quiz() — bonus feature
│                                   #  - summarize_notes() — bonus feature
│
└── data/
    └── sample_notes/
        ├── physics_newtons_laws.txt
        └── cs_binary_search_trees.txt
```

---

## 🧠 RAG Pipeline — Deep Dive

The RAG pipeline has three phases:

| Phase | What Happens | Technology |
|---|---|---|
| **Retrieval** | Embed question → ANN search in Endee → get top-K chunks | Endee + sentence-transformers |
| **Augmentation** | Inject retrieved chunks into LLM system prompt as context | Python string formatting |
| **Generation** | LLM generates answer strictly from provided context | Claude (claude-sonnet-4) |

### Why Endee for Retrieval?
- **Fast ANN search** using HNSW indexing
- **Cosine similarity** — perfect for normalized text embeddings
- **Metadata storage** — we store chunk text alongside vectors, so we can display source citations
- **REST API** — clean HTTP interface, easy to integrate
- **Local deployment** — data stays on your machine

### Chunking Strategy
Notes are split into 400-word chunks with 80-word overlap. The overlap ensures that sentences at chunk boundaries are captured in at least one chunk's context, preventing information loss at splits.

### Hallucination Prevention
The LLM system prompt strictly instructs Claude:
> *"If the answer is NOT found in the notes, say: 'This topic doesn't appear to be covered in your uploaded notes.'"*

This ensures the bot never makes up facts beyond your actual study material.

---

## ✨ Features Overview

| Feature | Description |
|---|---|
| PDF & Text Upload | Supports `.pdf`, `.txt`, `.md` files |
| Semantic Search | Endee cosine-similarity ANN search across all chunks |
| RAG Q&A | Claude answers grounded strictly in your notes |
| Source Citations | See which chunk of your notes was used to answer |
| Multi-turn Chat | Conversation memory across multiple questions |
| Auto Quiz Generator | MCQs generated from your notes via RAG |
| Notes Summarizer | Key concepts, definitions, formulas extracted |
| Multiple Files | Switch between different uploaded notes |

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Vector Database | **Endee** (endee-io/endee) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384 dim) |
| LLM | Anthropic Claude (`claude-sonnet-4`) |
| PDF Parsing | PyMuPDF (fitz) |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |

---

## 🌟 Endee Repository

As per the project requirements:

- ⭐ **Starred**: [endee-io/endee](https://github.com/endee-io/endee)
- 🍴 **Forked**: [YOUR_USERNAME/endee](https://github.com/YOUR_USERNAME/endee)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built as part of the Internship Project Evaluation — demonstrating practical RAG pipeline implementation using Endee Vector Database.*
