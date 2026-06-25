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

## Authentication options

The app is currently open — anyone with the URL can add/delete documents and query the index. Three realistic options, simplest first.

### Option 1 — HTTP Basic Auth (10 minutes, zero dependencies)

FastAPI has `HTTPBasic` built in. Add a single dependency middleware that checks every request against a hardcoded or env-var username/password. No sign-in UI needed — the browser shows a native prompt.

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets, os

security = HTTPBasic()

def require_auth(credentials: HTTPBasicCredentials = Depends(security)):
    ok = (
        secrets.compare_digest(credentials.username, os.environ["AUTH_USER"]) and
        secrets.compare_digest(credentials.password, os.environ["AUTH_PASS"])
    )
    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            headers={"WWW-Authenticate": "Basic"})
```

Add `Depends(require_auth)` to every route, or apply it globally via a middleware. Set `AUTH_USER` / `AUTH_PASS` as App Runner environment variables.

**Good for:** locking down a single-user demo quickly.  
**Not good for:** multiple users, sign-up flows, or anything public-facing.

---

### Option 2 — Clerk (recommended for multi-user)

[Clerk](https://clerk.com) is a hosted auth provider with a generous free tier (10k MAU). It handles sign-in UI, JWTs, session management, and social logins out of the box.

**How it fits this app:**

1. Add the Clerk JS SDK to `index.html` — it injects a `<SignIn>` component and attaches a JWT to every fetch automatically.
2. Verify the JWT in FastAPI using Clerk's public key:

```python
from clerk_backend_api import Clerk
clerk = Clerk(bearer_auth=os.environ["CLERK_SECRET_KEY"])

async def require_auth(request: Request):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    payload = clerk.verify_token(token)   # raises on invalid/expired
    return payload
```

3. Set `CLERK_PUBLISHABLE_KEY` (frontend) and `CLERK_SECRET_KEY` (backend) as App Runner env vars.

HTMX works fine with Clerk because HTMX sends the same headers as fetch — just add `hx-headers='{"Authorization": "Bearer <token>"}'` to the `<body>` tag and Clerk's JS fills in the token automatically.

**Good for:** sharing the demo with a small team, sign-up self-service, social logins (Google, GitHub).  
**Effort:** ~2 hours.  
**Cost:** free up to 10k MAU.

---

### Option 3 — IP allowlist at App Runner level (no code)

App Runner supports VPC ingress — restrict the service to a specific IP or CIDR via a VPC connector + security group, without touching the app code at all.

**Good for:** restricting to an office IP or personal IP while the app stays simple.  
**Not good for:** anyone who needs access from multiple locations.

---

### Recommendation

For a demo being shared with a small number of people: **Option 2 (Clerk)**. It's production-grade, takes an afternoon, and the free tier covers this use case entirely. Option 1 is fine for a temporary lock while Clerk is being set up.

---

## Anonymous read / signed-in write

### Concept

Let anyone query the shared corpus without signing in — the app is immediately useful as a demo. Signing in unlocks document management (upload, add text, delete) and access to your own private documents.

### Access tiers

| Action | Anonymous | Signed in |
|--------|-----------|-----------|
| Query against shared docs | ✓ | ✓ |
| Query against own private docs | — | ✓ |
| View shared document list | ✓ (read-only) | ✓ |
| Upload / add text | — | ✓ (private by default) |
| Delete own documents | — | ✓ |
| Re-index corpus.txt / chunk settings | — | admin only |
| Clear index | — | admin only |

### Backend changes

`require_auth` becomes optional — returns `None` for unauthenticated requests instead of raising 401. A separate `require_auth_strict` raises 401 for endpoints that truly need a user.

```python
async def optional_auth(request: Request) -> dict | None:
    if not ENABLED:
        return None
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None   # anonymous — not an error
    try:
        return _verify(auth.removeprefix("Bearer "))
    except Exception:
        raise HTTPException(401, "Invalid token")
```

**`/query`** uses `optional_auth`. The filter changes based on who's asking:
```python
uid = (auth or {}).get("sub")          # None for anonymous
if uid:
    src_filter = lambda doc: doc.metadata.get("visibility") in ("shared", uid)
else:
    src_filter = lambda doc: doc.metadata.get("visibility") == "shared"
```

**Write endpoints** (`/upload`, `/documents` POST, `/reindex`, `/rechunk`, `/rebuild`, `/clear`, `/save`, `/delete`) use `require_auth_strict` — 401 for anonymous.

### Frontend changes

Replace the full-page sign-in overlay with a non-blocking approach:

- The query panel is always visible and functional for anonymous users.
- A subtle banner at the top: `"Querying shared knowledge base — Sign in to add your own documents"` with a sign-in link. The banner disappears after sign-in.
- The corpus management panel (left column) shows shared docs to everyone. The "Manage corpus" details section is replaced with `"Sign in to upload or add documents"` for anonymous users.
- HTMX management buttons (upload, add text, delete) are hidden for anonymous users via a Jinja2 `{% if not anon %}` flag passed from the server, not just JavaScript — so they're never in the DOM at all for anonymous sessions.

### UX flow

1. User lands on the page → sees the shared corpus, can type a question and get an answer immediately.
2. They want to upload their own PDF → click "Sign in" → Clerk sign-in modal appears (not full-page, just the modal component).
3. After sign-in, the banner disappears, the corpus management panel unlocks, and their private documents appear in the doc list.
4. Their private documents are only searched when they are signed in.

### Why this is better than the current full-page gate

The current approach blocks all access until sign-in. That makes the app useless as a shareable demo — anyone you send the URL to hits a login wall before seeing anything. The anonymous-read model lets the shared corpus act as a live demo that sells the product before asking for a commitment.

---

## Multi-user data model — private and shared documents

### Current state

All signed-in users share one global `state.store`. Any document uploaded by one user is visible to all others and counts against the same index. There is no isolation.

### Proposed design — two visibility tiers

Add a `visibility` field to every document's metadata: either `"shared"` (visible to all) or the Clerk user ID (e.g. `"user_2abc..."`) for private documents. The query filter includes both the calling user's ID and `"shared"`.

**Storage:** No structural change to turbovec — metadata is already a free-form dict per chunk. Just store `{"source": ..., "chunk": ..., "visibility": "shared" | user_id}`.

**Query change** (`/query`):
```python
uid = auth_payload["sub"]   # Clerk user ID from JWT
src_filter = lambda doc: doc.metadata.get("visibility") in ("shared", uid)
# Combined with any source filter the user applied
```

**Upload:** Add a "Make public" checkbox (unchecked by default). Pass `visibility` in the form body; routes read it from `require_auth` return value for the user ID.

**Doc list:** Show shared docs with a globe icon; private docs are only visible to their owner. Admin users (identified by a Clerk role or a hardcoded user ID list) can see and delete all docs.

### Shared corpus (`corpus.txt`)

The existing `corpus.txt` reindex button makes sense as the admin-only "shared knowledge base" path. Tag all corpus chunks `visibility="shared"`. Only users with the admin role can re-index it.

### Storage implications

A single `TurboQuantVectorStore` works fine — turbovec already supports per-vector metadata and filtered search. No need for per-user index objects. Memory scales with total document count, not user count.

### Implementation steps (when needed)

1. Pass `auth_payload` into route handlers (change `require_auth` return type to include `sub`)
2. Add `visibility` to `chunk_with_meta()` signature
3. Update `/documents`, `/upload`, `/reindex` to set metadata
4. Update `/query` filter logic
5. Update doc list UI — filter server-side to caller's docs + shared
6. Add admin role check for corpus reindex and shared upload

---

## Document size limits

Currently no limits are enforced. A large upload can exhaust the 4 GB App Runner instance or tie up the event loop for minutes.

### Recommended limits

| Boundary | Limit | Reason |
|----------|-------|--------|
| Upload file size | 10 MB | Covers most PDFs; beyond this OCR + embedding takes minutes |
| Extracted text per document | 500 KB (~500 pages) | Caps chunk count at ~1 000 chunks/doc |
| Chunks per source | 500 | Prevents one source from dominating the index |
| Total index RAM | 80% of available | Already tracked via `psutil`; reject uploads when low |

### Implementation

In `/upload`, check immediately after `await file.read()`:
```python
if len(data) > 10 * 1024 * 1024:
    return HTMLResponse("File exceeds 10 MB limit.", status_code=413)
```

After text extraction, check extracted length:
```python
if len(content) > 500_000:
    content = content[:500_000]  # or reject with 413
```

Cap chunks:
```python
if len(chunks) > 500:
    chunks, metas = chunks[:500], metas[:500]
```

None of these require new dependencies.

---

## Rate limiting

Currently unlimited. A user can flood `/query` (each call = 1 Claude invocation + BGE embedding) or `/upload` with large files. With Clerk auth the risk is lower (only signed-in users), but still worth capping.

### Recommended per-user limits

| Endpoint | Limit | Reason |
|----------|-------|--------|
| `POST /query` | 10 req/min | Each query = Claude API call (cost + latency) |
| `POST /upload` | 5 req/min | OCR + embedding is CPU-heavy |
| `POST /documents` | 20 req/min | Text add is lighter but still runs the embedder |
| All others | 30 req/min | Corpus management, saves, deletes |

### Implementation options

**Option A — in-process token bucket (no dependencies)**

A module-level dict tracks `{user_id: (request_count, window_start)}`. Reset the count every 60 seconds. One function, ~15 lines, works perfectly for a single App Runner instance.

```python
import time
_rate: dict[str, tuple[int, float]] = {}

def check_rate(user_id: str, limit: int) -> bool:
    count, start = _rate.get(user_id, (0, time.monotonic()))
    if time.monotonic() - start > 60:
        count, start = 0, time.monotonic()
    _rate[user_id] = (count + 1, start)
    return count < limit
```

Call from each protected route and raise `HTTPException(429)` if it returns False. No Redis, no extra packages.

**Option B — `slowapi` library**

Drop-in FastAPI middleware backed by in-memory storage (or Redis for multi-instance). Cleaner if limits need to differ per route:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=lambda req: req.state.user_id)

@protected.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, ...):
    ...
```

Option A is recommended for now — it's sufficient for a single App Runner instance and adds zero dependencies.

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
| App Runner ARN | `arn:aws:apprunner:us-east-1:182879431700:service/turbovec-rag/33d5fdef7f10488bb219ab7b481f239c` |
| Service URL | `https://wzexxa5k5h.us-east-1.awsapprunner.com` |
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
