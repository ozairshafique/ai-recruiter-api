---
title: AI Recruiter API
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
---

# 🤖 AI Recruiter API

<div align="center">

![CI/CD Pipeline](https://github.com/ozairshafique/ai-recruiter-api/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-F7DF1E?logoColor=black)
![HuggingFace](https://img.shields.io/badge/API-HuggingFace-FFD21E?logo=huggingface&logoColor=black)
![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?logo=vercel&logoColor=white)

**Production-grade full-stack RAG system for intelligent CV analysis and candidate querying**

[🎨 Live Frontend](https://ai-recruiter-frontend-khaki.vercel.app) · [🚀 Live API](https://huggingface.co/spaces/ozair1112/ai-recruiter-api) · [📖 API Docs](https://ozair1112-ai-recruiter-api.hf.space/docs) · [💼 Portfolio](https://uzairshafique.vercel.app)

> **Note:** Live demo runs on HuggingFace free tier. If the API shows offline, wait 30 seconds and refresh.

</div>

---

## What It Does

AI Recruiter API is a production-ready full-stack RAG system that lets recruiters upload candidate CVs and query them using natural language. It automatically extracts text from PDFs, generates multilingual embeddings with Cohere, stores semantic vectors in FAISS, and retrieves answers using LangChain + LLaMA 3.3 70B via Groq — all within a fully containerized, CI/CD-deployed pipeline with a React frontend.

**Example:** _"Which candidate has the strongest Python and Machine Learning experience?"_
→ Structured answer with source attribution, page references, similarity scores, and latency tracking.

---

## System Architecture

<div align="center">
  <img src="images/architecturediagrams.png" alt="System Architecture — AI Recruiter API" width="100%"/>
</div>

<br/>

| Flow       | Steps                                                                         |
| ---------- | ----------------------------------------------------------------------------- |
| **Upload** | PDF → text extraction → chunking → Cohere embeddings → FAISS index            |
| **Query**  | Natural language → FAISS retrieval → LangChain RAG → LLaMA 3.3 70B → response |

LangSmith provides full LLMOps observability across both flows. DeepEval validates output quality with four RAG-specific metrics.

> **Vector search:** FAISS uses **cosine similarity** on Cohere's normalized embeddings — captures semantic meaning regardless of document length. Euclidean distance is not used as it's less effective for NLP retrieval tasks.

---

## Layer Overview

<div align="center">
  <img src="images/layeroverviews.png" alt="Layer Overview — AI Recruiter API" width="100%"/>
</div>

---

## CI/CD Pipeline

<div align="center">
  <img src="images/cicddiagrams.png" alt="CI/CD Pipeline — AI Recruiter API" width="60%"/>
</div>

<br/>

```
Developer pushes code
        ↓
Stage 1 — Run Tests (pytest)       ← fails fast, blocks next stage
        ↓
Stage 2 — Build Docker Image       ← docker build + compose validation
        ↓
Stage 3 — Deploy to HuggingFace    ← pushes to Spaces automatically
        ↓
Stage 4 — Health Check             ← GET /api/v1/health must return 200
        ↓
Live on HuggingFace Spaces ✅
```

Any stage failure stops the pipeline immediately and routes back to the developer with a clear error.

---

## Tech Stack

| Layer            | Technology                     |
| ---------------- | ------------------------------ |
| API framework    | FastAPI + Pydantic             |
| Frontend         | React 18 + Axios               |
| LLM              | LLaMA 3.3 70B via Groq         |
| Embeddings       | Cohere embed-multilingual-v3.0 |
| Vector store     | FAISS (cosine similarity)      |
| RAG framework    | LangChain + LangGraph          |
| Evaluation       | DeepEval                       |
| Observability    | LangSmith                      |
| Containerization | Docker + Docker Compose        |
| CI/CD            | GitHub Actions                 |
| Deployment       | HuggingFace Spaces + Vercel    |

---

## RAG Evaluation Results

Evaluated with DeepEval — all metrics exceed the 0.7 threshold:

| Metric               | Score | Threshold | Status  |
| -------------------- | ----- | --------- | ------- |
| Answer relevancy     | 0.94  | 0.7       | ✅ Pass |
| Faithfulness         | 0.91  | 0.7       | ✅ Pass |
| Contextual recall    | 0.89  | 0.7       | ✅ Pass |
| Contextual precision | 0.92  | 0.7       | ✅ Pass |

```bash
pytest tests/evaluation/test_rag_evaluation.py -s -v
```

---

## API Endpoints

| Method | Endpoint            | Description                  |
| ------ | ------------------- | ---------------------------- |
| `GET`  | `/`                 | Root endpoint                |
| `GET`  | `/api/v1/health`    | Health check                 |
| `POST` | `/api/v1/upload`    | Upload and index a CV (PDF)  |
| `POST` | `/api/v1/query`     | Query across all indexed CVs |
| `GET`  | `/api/v1/documents` | List indexed documents       |

Interactive docs: [/docs](https://ozair1112-ai-recruiter-api.hf.space/docs) · [/redoc](https://ozair1112-ai-recruiter-api.hf.space/redoc)

---

## Quick Start

### Prerequisites

- Python 3.12+
- Docker + Docker Compose
- [Groq API key](https://console.groq.com) — free tier available
- [Cohere API key](https://cohere.com) — free trial available

### Local setup

```bash
# Clone and enter the repo
git clone https://github.com/ozairshafique/ai-recruiter-api
cd ai-recruiter-api

# Create and activate virtual environment
python -m venv envs
source envs/bin/activate        # macOS/Linux
# envs\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API keys to .env

# Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker setup

```bash
docker-compose up --build -d
curl http://localhost:8000/api/v1/health
```

### Frontend setup

```bash
cd frontend
npm install
npm start
# Opens at http://localhost:3000
```

---

## Usage Examples

### Upload a CV

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@candidate_cv.pdf;type=application/pdf"
```

```json
{
  "status": "success",
  "filename": "candidate_cv.pdf",
  "document_id": "8011f755-...",
  "chunks": 12,
  "latency_ms": 342.5
}
```

### Query candidates

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which candidate has the most Python experience?", "top_k": 5}'
```

```json
{
  "status": "success",
  "answer": "Based on the indexed CVs, Candidate A has the strongest Python background...",
  "sources": [...],
  "model": "llama-3.3-70b-versatile",
  "latency": 661.09
}
```

---

## Environment Variables

```env
GROQ_API_KEY=your_groq_key
COHERE_API_KEY=your_cohere_key
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=AI-Recruiter-API
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
FAISS_INDEX_PATH=./faiss_index
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
```

---

## Project Structure

```
ai-recruiter-api/
│
├── 📁 app/                                 ← Application source code
│   ├── 📁 api/
│   │   └── 📄 routes.py                    ← All API endpoints
│   ├── 📁 core/
│   │   ├── 📄 config.py                    ← Settings & environment variables
│   │   └── 📄 logging.py                   ← Structured logging setup
│   └── 📁 services/
│       ├── 📄 embeddings.py                ← Cohere embed-multilingual-v3.0 + FAISS
│       ├── 📄 ingestion.py                 ← PDF loading, chunking (chunk=500, overlap=50)
│       ├── 📄 llm_service.py               ← LLaMA 3.3 70B via Groq API
│       ├── 📄 rag_pipeline.py              ← LangChain RAG orchestration + LangGraph
│       └── 📄 retriever.py                 ← FAISS similarity search (top-k)
│
├── 📁 frontend/                            ← React frontend (deployed on Vercel)
│   ├── 📁 src/
│   │   ├── 📄 App.js                       ← Layout, sidebar, routing
│   │   └── 📁 pages/
│   │       ├── 📄 Upload.js                ← PDF upload + indexing
│   │       ├── 📄 Query.js                 ← Natural language search
│   │       ├── 📄 Documents.js             ← Indexed CV list
│   │       └── 📄 Analytics.js             ← DeepEval scores dashboard
│   └── 📄 package.json
│
├── 📁 tests/
│   ├── 📄 conftest.py                      ← Shared pytest fixtures
│   ├── 📄 test_routes.py                   ← API endpoint tests
│   ├── 📄 test_rag_pipelines.py            ← Service unit tests
│   └── 📁 evaluation/
│       └── 📄 test_rag_evaluation.py       ← DeepEval RAG metrics
│
├── 📁 .github/workflows/
│   └── 📄 ci.yml                           ← CI/CD: test → build → deploy
│
├── 📁 images/                              ← Project diagrams (GitHub only)
│   ├── 🖼️  architecturediagrams.png
│   ├── 🖼️  cicddiagrams.png
│   └── 🖼️  layeroverviews.png
│
├── 📁 faiss_index/                         ← Vector store (git-ignored, auto-built)
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── ⚙️  pytest.ini
├── 🔒 .env.example
├── 🚫 .gitignore
├── 🚫 .dockerignore
└── 📄 requirements.txt
```

---

## Running Tests

```bash
# All tests (excluding evaluation)
pytest tests/ -v --ignore=tests/evaluation

# With coverage report
pytest tests/ -v --ignore=tests/evaluation --cov=app

# Evaluation suite (requires API keys — uses tokens)
pytest tests/evaluation/ -s -v
```

---

## Live Demo

| Resource         | Link                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------ |
| 🎨 Live Frontend | [ai-recruiter-frontend-khaki.vercel.app](https://ai-recruiter-frontend-khaki.vercel.app)                     |
| 🚀 Live API      | [huggingface.co/spaces/ozair1112/ai-recruiter-api](https://huggingface.co/spaces/ozair1112/ai-recruiter-api) |
| 📖 Swagger UI    | [/docs](https://ozair1112-ai-recruiter-api.hf.space/docs)                                                    |
| 📄 ReDoc         | [/redoc](https://ozair1112-ai-recruiter-api.hf.space/redoc)                                                  |

---

## Contributing

PRs and issues are welcome. Please open an issue before submitting large changes.

```bash
pytest tests/ -v --ignore=tests/evaluation --cov=app
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Uzair Shafique** — AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-uzairshafique-0077B5?logo=linkedin&logoColor=white)](https://linkedin.com/in/uzairshafique)
[![GitHub](https://img.shields.io/badge/GitHub-ozairshafique-181717?logo=github&logoColor=white)](https://github.com/ozairshafique)
[![Portfolio](https://img.shields.io/badge/Portfolio-uzairshafique.vercel.app-000000?logo=vercel&logoColor=white)](https://uzairshafique.vercel.app)

---

<div align="center">
  <sub>Built with FastAPI · React · LangChain · FAISS · Cohere · Groq · Docker · GitHub Actions</sub>
</div>
