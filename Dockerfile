FROM python:3.11-slim

# Install system dependencies for Postgres wheels and small Python build steps.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Docker Spaces run best with a normal non-root user.
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers

RUN mkdir -p "$HF_HOME" "$SENTENCE_TRANSFORMERS_HOME" \
    && chown -R user:user /home/user/.cache

WORKDIR /app

# Install uv for fast dependency management.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies before copying source so Docker can cache this layer.
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Copy the app code and the BM25 encoder file used for hybrid search.
COPY --chown=user . .

USER user

# Hugging Face Docker Spaces expose port 7860 by default.
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
