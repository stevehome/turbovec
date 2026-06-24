# ---- Build stage: compile the Rust/PyO3 extension -------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
ENV PATH="/root/.cargo/bin:$PATH"

RUN pip install --no-cache-dir "maturin[patchelf]"

WORKDIR /build

# Copy Cargo workspace (root manifests + both crates + build config)
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ .cargo/
COPY turbovec/ turbovec/
COPY turbovec-python/Cargo.toml turbovec-python/Cargo.toml
COPY turbovec-python/build.rs turbovec-python/build.rs
COPY turbovec-python/src/ turbovec-python/src/
COPY turbovec-python/python/ turbovec-python/python/
COPY turbovec-python/pyproject.toml turbovec-python/pyproject.toml
COPY turbovec-python/README.md turbovec-python/README.md

RUN cd turbovec-python && maturin build --release -o /dist/

# ---- Runtime stage ---------------------------------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the compiled wheel
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Install web app runtime dependencies
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" jinja2 python-multipart \
    "sentence-transformers>=3" \
    "langchain-anthropic>=0.3" "langchain-core>=0.3" \
    langchain-text-splitters pypdf psutil python-dotenv

# Pre-download models — avoids cold-start HuggingFace fetch
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the app (corpus, templates; saved_index excluded via .dockerignore)
COPY turbovec-python/app /app/app

EXPOSE 8000
CMD ["python", "app/server.py"]
