# 🤖 AI Recruiter API

![CI/CD Pipeline](https://github.com/ozairshafique/ai-recruiter-api/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Production-grade RAG API for intelligent CV analysis and candidate querying — built with FastAPI, LangChain, FAISS, Cohere embeddings, and LLaMA 3.3 70B via Groq.

---

## Overview

AI Recruiter API is a fully containerized RAG pipeline that allows recruiters to upload multiple CVs and query them using natural language. The API retrieves the most relevant chunks from indexed documents and generates accurate answers using LLaMA 3.3 70B.

---

## Features

- PDF upload and indexing with FAISS vector store
- Natural language querying across multiple CVs
- Multi-document retrieval with source attribution
- Full RAG pipeline with LangChain + LangGraph
- DeepEval evaluation (answer relevancy, faithfulness, contextual recall, contextual precision)
- LangSmith tracing for LLMOps observability
- Production-ready Docker + Docker Compose setup
- CI/CD with GitHub Actions
- Deployed on HuggingFace Spaces

---

## Tech Stack

| Layer            | Technology                     |
| ---------------- | ------------------------------ |
| API framework    | FastAPI + Pydantic             |
| LLM              | LLaMA 3.3 70B via Groq         |
| Embeddings       | Cohere embed-multilingual-v3.0 |
| Vector store     | FAISS                          |
| RAG framework    | LangChain + LangGraph          |
| Evaluation       | DeepEval                       |
| Observability    | LangSmith                      |
| Containerization | Docker + Docker Compose        |
| CI/CD            | GitHub Actions                 |
| Deployment       | HuggingFace Spaces             |

---

## API Endpoints

| Method | Endpoint            | Description                |
| ------ | ------------------- | -------------------------- |
| `GET`  | `/`                 | Root endpoint              |
| `GET`  | `/api/v1/health`    | Health check               |
| `POST` | `/api/v1/upload`    | Upload and index a CV      |
| `POST` | `/api/v1/query`     | Query across indexed CVs   |
| `GET`  | `/api/v1/documents` | List all indexed documents |

---

## Project Structure

```
ai-recruiter-api/
├── app/
│   ├── api/
│   │   └── routes.py              ← API endpoints
│   ├── core/
│   │   ├── config.py              ← settings
│   │   └── logging.py             ← logging setup
│   └── services/
│       ├── embeddings.py          ← FAISS + Cohere
│       ├── ingestion.py           ← PDF processing
│       ├── llm_service.py         ← LLaMA 3.3 70B
│       ├── rag_pipeline.py        ← RAG orchestration
│       └── retriever.py           ← similarity search
├── tests/
│   ├── conftest.py                ← shared fixtures
│   ├── test_routes.py             ← API endpoint tests
│   ├── test_rag_pipelines.py      ← service unit tests
│   └── evaluation/
│       └── test_rag_evaluation.py ← DeepEval metrics
├── .github/
│   └── workflows/
│       └── ci.yml                 ← CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Docker + Docker Compose
- Groq API key — [console.groq.com](https://console.groq.com)
- Cohere API key — [cohere.com](https://cohere.com)

### Local setup

```bash
# Clone the repo
git clone https://github.com/ozairshafique/ai-recruiter-api
cd ai-recruiter-api

# Create virtual environment
python -m venv envs
source envs/bin/activate  # Windows: envs\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Add your API keys

# Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker setup

```bash
# Run with Docker Compose
docker-compose up --build -d

# Check health
curl http://localhost:8000/api/v1/health
```

---

## Environment Variables

Create a `.env` file in the project root:

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

## Usage

### Upload a CV

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@candidate_cv.pdf;type=application/pdf"
```

Response:

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

Response:

```json
{
  "status": "success",
  "answer": "The candidate has strong Python experience...",
  "sources": [...],
  "model": "llama-3.3-70b-versatile",
  "latency": 661.09
}
```

---

## RAG Evaluation

Evaluation suite using DeepEval with real LLM metrics:

```bash
pytest tests/evaluation/test_rag_evaluation.py -s -v
```

| Metric               | Score | Threshold |
| -------------------- | ----- | --------- |
| Answer relevancy     | 0.94  | 0.7       |
| Faithfulness         | 0.91  | 0.7       |
| Contextual recall    | 0.89  | 0.7       |
| Contextual precision | 0.92  | 0.7       |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v --ignore=tests/evaluation

# Run with coverage
pytest tests/ -v --ignore=tests/evaluation --cov=app

# Run evaluation tests (requires API keys + tokens)
pytest tests/evaluation/ -s -v
```

---

## CI/CD Pipeline

GitHub Actions pipeline runs automatically on every push to `main`:

```
Push to main
     ↓
Run Tests (pytest)          ← always runs
     ↓
Build Docker Image          ← only on main
     ↓
Deploy to HuggingFace       ← only on main
```

---

## Live Demo

API live at: [huggingface.co/spaces/ozair1112/ai-recruiter-api](https://huggingface.co/spaces/ozair1112/ai-recruiter-api)

Interactive docs: [/docs](https://ozair1112-ai-recruiter-api.hf.space/docs)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Uzair Shafique** — AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-uzairshafique-0077B5?logo=linkedin)](https://linkedin.com/in/uzairshafique)
[![GitHub](https://img.shields.io/badge/GitHub-ozairshafique-181717?logo=github)](https://github.com/ozairshafique)
[![Portfolio](https://img.shields.io/badge/Portfolio-uzairshafique.vercel.app-black)](https://uzairshafique.vercel.app)
