---
title: News Chatbot
emoji: 📰
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
short_description: FastAPI news chatbot over selectable indexed news sources
---

# News Chatbot

FastAPI chatbot for searching and answering questions over selectable indexed news sources, including Sakal and Bar & Bench.

## Hugging Face Spaces

This app is configured as a Docker Space. Hugging Face will build the `Dockerfile` and run the FastAPI app on port `7860`.

The hosted app expects these runtime secrets:

- `POSTGRESQL_URL`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_HOST`
- `DEFAULT_NEWS_SOURCE`
- `BARANDBENCH_PINECONE_INDEX_HOST`
- `BARANDBENCH_PINECONE_NAMESPACE`
- `BARANDBENCH_BM25_ENCODER_PATH`
- `SAKAL_PINECONE_INDEX_HOST`
- `SAKAL_PINECONE_NAMESPACE`
- `SAKAL_BM25_ENCODER_PATH`

Optional search settings:

- `PINECONE_NAMESPACE` defaults to `default`
- `BM25_ENCODER_PATH` defaults to `bm25_sakal_values.json`

For direct OpenAI, set:

- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL`

For Azure OpenAI, set:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_EMBEDDING_API_VERSION`
- `AZURE_OPENAI_CHAT_API_VERSION`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`

Ingestion is not run on the hosted app. The hosted app only queries existing Pinecone indexes and source-specific BM25 encoder files.
