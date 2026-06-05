# News Chatbot POC Overview

## Description

The News Chatbot is a proof-of-concept application that allows users to ask questions in natural language and receive answers based on indexed news content. Instead of generating answers only from model knowledge, the application retrieves relevant news content first and then produces a response grounded in those sources.

The current POC uses content from the following news data sources:

- Bar & Bench
- eSakal

## POC Requirements

The News Chatbot POC was expected to demonstrate the core capabilities of a retrieval-based conversational assistant for news content.

### Functional Requirements

- Accept natural language questions from users through a chat-style interface.
- Search across indexed news content to find relevant articles or article snippets from Bar & Bench and eSakal.
- Generate responses grounded in retrieved source material instead of relying only on model knowledge.
- Present answers in a concise and readable format.
- Include source references so users can understand where the answer came from.
- Support follow-up questions within the same conversation context.

### Non-Functional Requirements

- Keep the solution modular so that additional news sources can be added later.
- Maintain a design that is suitable for experimentation and iteration during the POC stage.
- Allow the system to be run and tested locally through a simple API-based setup.
- Keep the implementation generic enough to adapt to different content sources or business domains in future phases.

## Approach Taken

The POC was implemented using a Retrieval-Augmented Generation (RAG) approach.

In simple terms, this means the chatbot does not try to answer from memory alone. Instead, it first searches the news data, collects the most relevant content, and then uses that content to generate a grounded response. This is useful for a news chatbot because news information changes frequently, and answers should stay tied to actual source material.

### Workflow

1. A user sends a question to the chatbot.
2. The system analyzes the question and prepares a search-friendly query.
3. The retrieval layer searches the indexed news dataset for relevant content.
4. The most useful results are collected as context.
5. The language model generates a final answer based on that retrieved context.
6. The response is returned along with source references.

## Testing and Evaluation

The POC also includes support for local testing and internal evaluation workflows.

- The application can be run locally and tested end to end through its browser-based interface and API endpoints.
- The project includes automated testing for chatbot behavior across multi-turn conversations.
- The evaluation approach includes custom benchmark-style testing for retrieval quality and answer quality.
- Retrieval evaluation is intended to measure how effectively the system finds the correct supporting content, using metrics such as hit rate, hit@1, hit@3, precision@3, recall@5, MRR, and nDCG@5.
- Response evaluation is intended to check relevance, coverage, coherence, and overall helpfulness.

In summary, the testing approach helps validate both major parts of the solution: retrieving the right news content and generating useful responses from that content.

## Architecture Diagram

![News Chatbot Architecture](https://drive.google.com/uc?export=view&id=1iRdrBrh15OJ44yR22nkPC0QisQQUZ0HN)
