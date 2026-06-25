# Project Review

## Structure

```
turbovec/                    Rust core crate (crates.io: turbovec)
  src/{lib,encode,search,pack,codebook,rotation,id_map,io,error}.rs
  tests/                     15 integration test files
  Cargo.toml

turbovec-python/             PyO3 wheel (PyPI: turbovec)
  src/lib.rs                 PyO3 glue — wraps Rust types, converts numpy arrays
  python/turbovec/
    __init__.py              re-exports TurboQuantIndex, IdMapIndex
    langchain.py  (558 ln)   LangChain VectorStore adapter
    llama_index.py           LlamaIndex adapter
    haystack.py              Haystack adapter
    agno.py                  Agno adapter
    _dedup.py / _persist.py  helpers
  app/                       ← demo app (NOT in the wheel)
    server.py    (505 ln)
    templates/index.html
    data/
  tests/                     Python integration tests
  pyproject.toml

examples/downstream-smoke/   standalone Cargo project, tests real downstream linking
benchmarks/suite/            self-contained Python benchmark scripts
docs/                        API docs, SVG charts, test PDFs
planning/                    PLAN.md, REVIEW.md
Dockerfile                   multi-stage; builder=Rust+maturin, runtime=Python+model
```

---

## Issues

**1. `pyproject.toml` ships server-only deps as hard requirements**

`psutil`, `pypdf`, `python-dotenv`, and `langchain-text-splitters` are listed under
`[project] dependencies` but are only used by `app/server.py` — not by the package itself.
Anyone who does `pip install turbovec` gets all four pulled in unnecessarily.

**2. `server.py` is 505 lines of mixed concerns**

Routes, OCR, enrichment, chunking, persistence, and HTML templating are all in one file.
Natural split: `ingest.py` (chunking / OCR / enrichment), routes stay in `server.py`.

**3. Global mutable state**

`_store`, `_sources`, `_chunk_size`, etc. are module-level globals mutated by multiple
endpoints. Safe with one Uvicorn worker, breaks silently under `--workers 2`.

**4. `_extract_text` / `_extract_text_with_claude` are synchronous**

They block the asyncio event loop during large PDF processing. Should be wrapped in
`loop.run_in_executor(None, ...)`.

**5. `sources.json` holds full text of every uploaded document**

A 10 MB PDF produces a 10 MB `sources.json`. Better to store each source as a separate
file under `data/sources/<filename>`.

**6. `plan/` and `planning/` both exist** — one directory is enough.

**7. `docs/` has personal test PDFs in git** — excluded from Docker but committed to the repo.

**8. No authentication** — if deployed publicly anyone can upload, delete, or clear the index.

---

## Clean separation — shipping without source

The goal: publish `turbovec` to crates.io and PyPI, then run the demo app by installing
from those registries, no source needed.

**Step 1 — Fix `pyproject.toml`**

Move server-only deps out of `[project] dependencies` into an `[app]` optional group:

```toml
[project]
dependencies = ["numpy>=1.20"]   # only true runtime dep of the package

[project.optional-dependencies]
langchain   = ["langchain-core>=0.3"]
llama-index = ["llama-index-core>=0.11"]
haystack    = ["haystack-ai>=2.0"]
agno        = ["agno>=2.0"]
app         = [
    "fastapi", "uvicorn", "jinja2", "python-multipart",
    "sentence-transformers", "langchain-anthropic",
    "langchain-core>=0.3", "langchain-text-splitters",
    "pypdf", "psutil", "python-dotenv", "anthropic",
]
```

**Step 2 — Move the demo app out of `turbovec-python/`**

```
turbovec-demo/        ← new top-level directory (or separate repo)
  pyproject.toml      depends on: turbovec[app]
  server.py
  templates/
  data/
  Dockerfile
  README.md
```

`turbovec-python/` then contains only the wheel — bindings + adapters, no app code.

**Step 3 — Publish**

```bash
# Rust crate (Ryan's credentials)
cargo publish -p turbovec

# Python wheel (Ryan's credentials)
cd turbovec-python && maturin publish
```

**Step 4 — Users run the demo with no source**

```bash
pip install "turbovec[app]"
# download server.py + templates/ from turbovec-demo repo
export ANTHROPIC_API_KEY=sk-...
uvicorn server:app --host 0.0.0.0 --port 8000
```

Or Docker (already works today):

```bash
docker build -t turbovec-rag .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... turbovec-rag
# opens at http://localhost:8000
```

---

## User instructions (current state, source required)

```bash
# 1. Clone
git clone https://github.com/stevehome/turbovec-demo
cd turbovec-demo

# 2. Build the Rust extension
cd turbovec-python
unset CONDA_PREFIX          # if conda is active
uv run maturin develop --release

# 3. Install app dependencies
uv pip install fastapi uvicorn jinja2 python-multipart \
    sentence-transformers langchain-anthropic "langchain-core>=0.3" \
    langchain-text-splitters pypdf psutil python-dotenv anthropic

# 4. Set API key
echo "ANTHROPIC_API_KEY=sk-..." > .env

# 5. Run
uv run python app/server.py
# opens at http://127.0.0.1:8000
```

---

## Deployment — AWS App Runner

### Infrastructure

| Resource | Value |
|----------|-------|
| ECR repo | `182879431700.dkr.ecr.us-east-1.amazonaws.com/turbovec-rag` |
| App Runner ARN | `arn:aws:apprunner:us-east-1:182879431700:service/turbovec-rag/9e332d0af45149dbaf6cc2f60891443a` |
| Service URL | `https://divqjjjhuh.us-east-1.awsapprunner.com` |
| Region | us-east-1 |
| Instance | 2 vCPU / 4 GB |
| ECR access role | `arn:aws:iam::182879431700:role/service-role/AppRunnerECRAccessRole` |
| AWS account | 182879431700 (user: aiengineer) |

### Deploy procedure

```bash
# 1. Build image
docker build -t turbovec-rag .

# 2. Authenticate with ECR
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    182879431700.dkr.ecr.us-east-1.amazonaws.com

# 3. Tag and push
docker tag turbovec-rag:latest \
  182879431700.dkr.ecr.us-east-1.amazonaws.com/turbovec-rag:latest
docker push \
  182879431700.dkr.ecr.us-east-1.amazonaws.com/turbovec-rag:latest

# 4. Trigger redeployment (after first service exists)
aws apprunner start-deployment \
  --service-arn arn:aws:apprunner:us-east-1:182879431700:service/turbovec-rag/59dbd5b073a64a78a85f48b65907dee7 \
  --region us-east-1

# 5. Check status
aws apprunner describe-service \
  --service-arn arn:aws:apprunner:us-east-1:182879431700:service/turbovec-rag/59dbd5b073a64a78a85f48b65907dee7 \
  --region us-east-1 \
  --query 'Service.{Status:Status,URL:ServiceUrl}' --output table
```

### First-time service creation

```bash
aws apprunner create-service \
  --service-name turbovec-rag \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "182879431700.dkr.ecr.us-east-1.amazonaws.com/turbovec-rag:latest",
      "ImageConfiguration": {
        "Port": "8000",
        "RuntimeEnvironmentVariables": {
          "ANTHROPIC_API_KEY": "<key>"
        }
      },
      "ImageRepositoryType": "ECR"
    },
    "AuthenticationConfiguration": {
      "AccessRoleArn": "arn:aws:iam::182879431700:role/service-role/AppRunnerECRAccessRole"
    }
  }' \
  --instance-configuration '{"Cpu": "2 vCPU", "Memory": "4 GB"}' \
  --health-check-configuration '{"Protocol": "HTTP", "Path": "/health", "Interval": 20, "Timeout": 10, "HealthyThreshold": 1, "UnhealthyThreshold": 5}' \
  --region us-east-1
```

### Known issue — startup health check failure

**Problem:** The first deployment failed with "Health check failed. Check your configured port number."

**Root cause:** All model loading (BGE-base-en-v1.5, cross-encoder, index) runs synchronously during Python module import in `store.py`, before uvicorn ever binds port 8000. App Runner fires health checks immediately after the container starts. Since uvicorn hasn't started yet, the health check gets connection refused and the deployment fails.

**Fix required (not yet implemented):**

1. Wrap all startup code in `store.py` in a `def initialize()` function — don't run at import time.
2. In `server.py`, use FastAPI `lifespan` to call `initialize()` after uvicorn binds the port:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(store.initialize)
    yield

app = FastAPI(title="turbovec RAG", lifespan=lifespan)
```

3. Add a `GET /health` endpoint that returns 200 immediately (before models are loaded).
4. Update App Runner health check path from `/` to `/health`.

With this fix, uvicorn binds port 8000 first, the health check passes immediately, and model loading completes in the background before any real requests are served.

---

## Deployment war story — building for App Runner on an M1/M2/M3 Mac

### What was implemented

The lifespan fix above was implemented (`store.initialize()`, `GET /health`). The real blocker turned out to be a completely different problem: **architecture mismatch**.

### The architecture problem

App Runner runs on x86_64. An M-series Mac builds Docker images for arm64 by default. The first two deployments pushed an arm64 image to ECR. App Runner pulled it, tried to start the container, the container exited immediately (wrong ELF architecture), and the health check got connection refused — same symptom as the previous bug, different cause.

There are no application logs at all when this happens (the CloudWatch log stream stays empty), which makes it look identical to the "uvicorn not started" issue.

### Cross-compilation attempts

`docker buildx build --platform linux/amd64` was the obvious fix. Three attempts:

**Attempt 1 — pure QEMU emulation** (no `FROM --platform=$BUILDPLATFORM`)  
QEMU runs the entire build — including Rust build scripts — as x86_64 binaries. Rust build scripts are arm64 ELF binaries. QEMU can't execute them.  
Result: `signal: 4, SIGILL: illegal instruction` in the `quote` build script.

**Attempt 2 — native builder, wrong packages**  
`FROM --platform=$BUILDPLATFORM` makes the builder stage run natively on arm64 (no QEMU). Added `gcc-x86-64-linux-gnu` for the cross-linker and `libopenblas-dev:amd64` for the x86_64 OpenBLAS that turbovec links against.  
Result: Rust compiled successfully. Linker failed: `cannot find crti.o` — the C runtime startup object for x86_64 wasn't installed.

**Attempt 3 — add `crossbuild-essential-amd64`**  
Replaced `gcc-x86-64-linux-gnu` with `crossbuild-essential-amd64` (the Debian meta-package that includes `libc6-dev:amd64` which provides `crti.o`). Also used `RUSTFLAGS="-C target-cpu=x86-64-v3 -L /usr/lib/x86_64-linux-gnu"`.  
Result: `RUSTFLAGS` applies to arm64 build scripts too. `-C target-cpu=x86-64-v3` is invalid on arm64. Build scripts for `quote`, `proc-macro2`, `libc` etc. all failed immediately.

**Attempt 4 — correct approach**  
`gcc-x86-64-linux-gnu` + `libc6-dev:amd64` + `libopenblas-dev:amd64`. Moved cross-linker and `-L` path into a per-target cargo config (`/root/.cargo/config.toml`) instead of `RUSTFLAGS`, so arm64 build scripts are completely unaffected. No env vars touching the compiler flags.  
Result: **Rust cross-compilation succeeded in 63 seconds.** But the subsequent step — `RUN python -c "... SentenceTransformer('BAAI/bge-base-en-v1.5') ..."` — hung for hours. HuggingFace model downloads inside a Docker build (where you can't see the progress) are unreliable; if the connection drops mid-download, the build hangs silently.

### Is it feasible on an M-series Mac?

**The Rust cross-compilation problem is solved.** The working Dockerfile approach:

```dockerfile
FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder

RUN dpkg --add-architecture amd64 && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config \
    gcc-x86-64-linux-gnu \
    libc6-dev:amd64 \
    libopenblas-dev:amd64 \
    && rm -rf /var/lib/apt/lists/*

RUN rustup target add x86_64-unknown-linux-gnu
# Per-target config — never touches arm64 build scripts
RUN mkdir -p /root/.cargo && printf \
    '[target.x86_64-unknown-linux-gnu]\nlinker = "x86_64-linux-gnu-gcc"\nrustflags = ["-L", "/usr/lib/x86_64-linux-gnu"]\n' \
    >> /root/.cargo/config.toml

RUN cd turbovec-python && maturin build --release -o /dist/ --target x86_64-unknown-linux-gnu
```

**The model pre-download step is the remaining blocker.** Baking HuggingFace models into the image during `docker build` is fragile — there's no retry, no progress display, and a stalled connection hangs silently for hours. Options:

1. **Remove the model bake-in step from the Dockerfile** and instead download models at container startup (first request is slow but deployment is reliable). Use `HF_HUB_OFFLINE=0` and accept the cold-start penalty, or mount a model cache volume.
2. **Pre-download to a local path and COPY into the image** — download models outside Docker, copy them in with `COPY`. Reliable but requires 500 MB of models on disk locally.
3. **Use a CI/CD runner on x86_64** — GitHub Actions `ubuntu-latest` runner builds natively, no cross-compilation needed. The workflow `.github/workflows/deploy.yml` is already committed and the IAM credentials are set as repo secrets. Blocked only by GitHub billing (`AKIASVFDWVQKAGCBT252` is the Actions IAM user, secrets set in stevehome/turbovec-demo).

### Recommended path forward

Fix the GitHub billing issue — go to github.com/settings/billing and verify payment method. Once Actions runs, every push to main automatically builds on x86_64, pushes to ECR, and triggers App Runner. No local Docker builds needed.

If billing can't be fixed quickly, use option 2 (local model download + COPY) to get a one-time working image built and deployed.

---

## Previous review (Phase 1 → Phase 2 transition)

See `plan/REVIEW.md` for the earlier review written before the FastAPI + HTMX app was
built — covers Gradio Phase 1 gaps and the professional frontend recommendation that led
to the current HTMX implementation.
