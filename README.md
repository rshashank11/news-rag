# NewsGPT

> A FastAPI-based hybrid RAG chatbot for searching and answering questions from indexed news archives.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

NewsGPT answers questions grounded exclusively in retrieved news content — it never draws on model memory. It supports two sources out of the box:

| Source | Language | Index |
|---|---|---|
| [Sakal](https://www.sakal.com/) | Marathi | Pinecone |
| [Bar & Bench](https://www.barandbench.com/) | English | Pinecone + PostgreSQL |

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Ingestion](#ingestion)
- [Tests](#tests)
- [Project Structure](#project-structure)

---

## How It Works

```
User question
     │
     ▼
 Planner — rewrites the question into a source-aware search query
     │
     ▼
 Retriever — dense embedding + BM25 sparse vectors → Pinecone hybrid search
     │
     ▼
 Context judge — checks whether retrieved chunks are sufficient
     │
     ▼
 Answer — generates a grounded response with inline citations
```

1. The **planner** turns the user's question into a source-aware search query.
2. The **retriever** embeds the query for semantic (dense) search.
3. The **BM25 encoder** converts the query into sparse keyword values.
4. **Pinecone** runs hybrid search combining dense and sparse vectors.
5. The **context judge** decides whether retrieved chunks are sufficient.
6. The **answer** step writes a source-grounded response with citations.

## Architecture

![NewsGPT Architecture](docs/newsgpt-architecture.svg)

The editable source diagram is at `docs/newsgpt-architecture.drawio`.

---

## Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or pip
- Docker & Docker Compose (for containerised runs and ingestion)
- A [Pinecone](https://www.pinecone.io/) account with indexes created for each source
- An OpenAI API key **or** an Azure OpenAI deployment

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-org/news-rag.git
cd news-rag
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in the required values (see Configuration below)
```

### 3. Start the API

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

| Endpoint | URL |
|---|---|
| API root | http://localhost:8000 |
| Interactive docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

### Docker Compose

```bash
docker compose up app
```

---

## Configuration

Copy `.env.example` to `.env` and fill in real values. **Never commit `.env`.**

### Required

| Variable | Description |
|---|---|
| `POSTGRESQL_URL` | PostgreSQL connection string |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_HOST` | Default Pinecone index host |
| `DEFAULT_NEWS_SOURCE` | `sakal` or `barandbench` |
| `OPENAI_API_KEY` | OpenAI API key (or use Azure settings below) |

### Source-specific

| Variable | Description |
|---|---|
| `BARANDBENCH_PINECONE_INDEX_HOST` | Bar & Bench index host |
| `BARANDBENCH_PINECONE_NAMESPACE` | Bar & Bench namespace |
| `BARANDBENCH_BM25_ENCODER_PATH` | Path to BM25 values JSON (default: `bm25_barandbench_values.json`) |
| `SAKAL_PINECONE_INDEX_HOST` | Sakal index host |
| `SAKAL_PINECONE_NAMESPACE` | Sakal namespace |
| `SAKAL_BM25_ENCODER_PATH` | Path to BM25 values JSON (default: `bm25_sakal_values.json`) |

### OpenAI

| Variable | Description |
|---|---|
| `OPENAI_CHAT_MODEL` | Chat completion model (e.g. `gpt-4o-mini`) |
| `OPENAI_EMBEDDING_MODEL` | Embedding model (e.g. `text-embedding-3-small`) |
| `OPENAI_CONTEXT_JUDGE_MODEL` | Model used for context sufficiency check |

### Azure OpenAI (alternative to direct OpenAI)

| Variable | Description |
|---|---|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint URL |
| `AZURE_OPENAI_API_VERSION` | API version |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding deployment name |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Chat deployment name |
| `AZURE_OPENAI_PLANNER_DEPLOYMENT` | Planner deployment name |
| `AZURE_OPENAI_CONTEXT_JUDGE_DEPLOYMENT` | Context judge deployment name |

### Retrieval (optional)

| Variable | Default | Description |
|---|---|---|
| `HYBRID_ALPHA` | `0.5` | Dense-vs-sparse balance (0 = sparse only, 1 = dense only) |
| `RETRIEVAL_TOP_K` | `10` | Number of chunks to retrieve |
| `RERANK_MODE` | `none` | `none` (Pinecone scores) or `jina` (Jina reranker) |
| `JINA_API_KEY` | — | Required when `RERANK_MODE=jina` |
| `MAX_CONTEXT_JUDGE_CHARS_PER_SOURCE` | `1500` | Max chars sent to the context judge per source |

---

## Ingestion

### Sakal

```bash
# 1. Convert XML dumps to JSON
python convert_sakal_xml_to_json.py

# 2. Chunk, embed, and upload to Pinecone
docker compose --profile ingest run ingest-sakal
```

### Bar & Bench

```bash
# Ingest story dumps into Postgres + Pinecone
docker compose --profile ingest run ingest-barandbench

# Backfill published_at metadata for existing Pinecone chunks (if needed)
python backfill_barandbench_pinecone_published_at.py
```

---

## Tests

```bash
uv run python -m unittest discover -s tests
```

---

## Project Structure

```
news-rag/
├── app/
│   ├── agents/          # LangGraph agent definitions
│   ├── config.py        # Settings (pydantic-settings)
│   ├── embeddings.py    # Embedding helpers
│   ├── retrieval.py     # Hybrid retrieval logic
│   ├── sparse.py        # BM25 sparse encoder
│   └── vectorstore.py   # Pinecone client wrapper
├── docs/                # Architecture diagrams
├── tests/               # Unit tests
├── main.py              # FastAPI application entry point
├── ingest_sakal.py      # Sakal ingestion script
├── ingest_barandbench.py# Bar & Bench ingestion script
├── .env.example         # Environment variable template
├── docker-compose.yml
└── pyproject.toml
```
