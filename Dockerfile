FROM python:3.11-slim

# Install system dependencies for Postgres and building AI tools
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

# Install uv for lightning-fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml .

# Install CPU-only PyTorch first. This avoids downloading huge CUDA/GPU wheels on CPU hosts.
RUN uv pip install --system "torch==2.5.1+cpu" --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies into the system environment.
RUN uv pip install --system -r pyproject.toml

# Copy the rest of your code
COPY --chown=user . .

USER user

# Hugging Face Docker Spaces expose port 7860 by default.
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
