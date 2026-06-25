# ---- Build stage: compile the Rust/PyO3 extension -------------------
# FROM --platform=$BUILDPLATFORM keeps the builder on the native host so
# Rust build scripts run without QEMU emulation. On amd64 CI (GitHub
# Actions) this is a no-op; on arm64 (Apple Silicon) it enables native
# cross-compilation to the amd64 target via LLVM.
FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder

# On arm64 hosts, install the amd64 cross-toolchain and x86_64 OpenBLAS.
# On amd64 hosts (CI), use the standard native toolchain.
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      dpkg --add-architecture amd64 && apt-get update && \
      apt-get install -y --no-install-recommends \
        curl build-essential pkg-config \
        gcc-x86-64-linux-gnu \
        libc6-dev:amd64 \
        libopenblas-dev:amd64; \
    else \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        curl build-essential pkg-config libopenblas-dev; \
    fi && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
ENV PATH="/root/.cargo/bin:$PATH"

RUN rustup target add x86_64-unknown-linux-gnu

# On arm64, configure the cross-linker per-target so RUSTFLAGS never
# touches arm64 build scripts. Not needed on amd64 (native toolchain).
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      mkdir -p /root/.cargo && printf \
        '[target.x86_64-unknown-linux-gnu]\nlinker = "x86_64-linux-gnu-gcc"\nrustflags = ["-L", "/usr/lib/x86_64-linux-gnu"]\n' \
        >> /root/.cargo/config.toml; \
    fi

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

RUN cd turbovec-python && maturin build --release -o /dist/ --target x86_64-unknown-linux-gnu

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
    langchain-text-splitters pypdf psutil python-dotenv \
    "PyJWT[crypto]" httpx

# Pre-download models — avoids cold-start HuggingFace fetch
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-base-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the app (corpus, templates; saved_index excluded via .dockerignore)
COPY turbovec-python/app /app/app

EXPOSE 8000
CMD ["python", "app/server.py"]
