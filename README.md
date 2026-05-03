---
title: News RAG
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# News RAG

FastAPI chatbot for searching and answering questions over ingested news stories.

The deployed app expects these runtime environment variables:

- `DATABASE_URL`
- `OPENSEARCH_URL`
- `OPENSEARCH_INDEX_NAME`
- `OPENAI_API_KEY`
- `JINA_API_KEY` optional
- `EMBEDDING_MODEL_NAME`
- `EMBEDDING_DEVICE`
- `NORMALIZE_EMBEDDINGS`

Ingestion is not run on the hosted app. The hosted app only queries the existing Postgres database and OpenSearch index.
