# Palm Mind RAG

Palm Mind RAG is a FastAPI backend for uploading documents, asking questions
about those documents, and booking interviews through a conversational API.

The application implements a custom Retrieval-Augmented Generation (RAG)
pipeline. It does not use `RetrievalQAChain`, FAISS, Chroma, or a frontend.

## What it can do

- Upload `.pdf` and UTF-8 `.txt` files.
- Extract text from PDFs and text files.
- Split text with either fixed-size or sentence-aware chunking.
- Generate 384-dimensional embeddings with FastEmbed.
- Store vectors in Qdrant and document metadata in PostgreSQL.
- Answer questions using retrieved document chunks and Groq.
- Keep multi-turn chat history in Redis.
- Rewrite follow-up questions before retrieval.
- Collect interview booking details conversationally:
  name, email, date, and time.
- Validate booking details and ask for confirmation.
- Store confirmed bookings in PostgreSQL.
- Run all services with Docker Compose.

## Architecture

```text
Client
  |
  v
FastAPI
  |
  +-- Document API
  |     +-- pypdf text extraction
  |     +-- fixed/sentence chunking
  |     +-- FastEmbed embeddings
  |     +-- PostgreSQL metadata
  |     +-- Qdrant vectors
  |
  +-- Chat API
        +-- Redis conversation memory
        +-- Booking state machine
        +-- Qdrant retrieval
        +-- Groq LLM answer generation
```

The Docker Compose stack contains:

| Service | Purpose | Local port |
|---|---|---:|
| `api` | FastAPI application | `8000` |
| `postgres` | Documents and bookings | `5432` |
| `redis` | Chat history and temporary booking state | `6379` |
| `qdrant` | Vector storage and similarity search | `6333` |

## Requirements

For the recommended Docker setup, install:

- Docker Desktop with Docker Compose
- A Groq API key

For running the application directly with Python, use Python 3.12 or newer.
The project is tested with Python 3.13 as well.

## Quick start with Docker

### 1. Create the environment file

From the repository root, run:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `GROQ_API_KEY` in `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Do not commit `.env`. It is ignored by Git. The committed
`.env.example` contains configuration defaults but no secret.

### 2. Build and start everything

```powershell
docker compose up --build
```

The first build downloads and embeds the FastEmbed model into the API image.
The initial build can therefore take longer than later starts.

To run in the background:

```powershell
docker compose up --build -d
```

### 3. Open the API documentation

Open [http://localhost:8000/docs](http://localhost:8000/docs) in a browser.
The Swagger UI lets you upload files and call every endpoint without writing
client code.

Check that all backing services are available:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

An application with healthy dependencies returns a report similar to:

```json
{
  "status": "ok",
  "services": {
    "postgres": "ok",
    "redis": "ok",
    "qdrant": "ok"
  }
}
```

If a dependency is unavailable, the endpoint returns HTTP `503` and identifies
the failing service.

## Using the API

The examples below use PowerShell. The same operations can also be performed
from the Swagger UI at `/docs`.

### Upload a document

Supported file types are `.pdf` and UTF-8 `.txt`. The maximum upload size is
10 MB.

Sentence-aware chunking is the default:

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/documents" `
  -F "file=@.\yourfile.pdf" `
  -F "strategy=sentence"
```

Use fixed-size chunks instead:

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/documents" `
  -F "file=@.\yourfile.pdf" `
  -F "strategy=fixed" `
  -F "chunk_size=800" `
  -F "chunk_overlap=100"
```

`chunk_size` must be at least `100`, and `chunk_overlap` must be smaller than
`chunk_size`.

The response contains the document ID, processing status, selected chunking
configuration, and number of chunks. Uploading the exact same file bytes again
does not create another copy; it returns HTTP `200` with `"duplicate": true`.

### List uploaded documents

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/documents
```

### Delete a document

Replace the ID with the `id` returned from the upload response:

```powershell
Invoke-WebRequest `
  -Method Delete `
  -Uri "http://localhost:8000/api/v1/documents/<document-id>"
```

Deletion removes the Qdrant vectors before deleting the PostgreSQL row.

### Ask a question

```powershell
$body = @{
  message = "What is the refund policy?"
} | ConvertTo-Json

$response = Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/v1/chat `
  -ContentType "application/json" `
  -Body $body

$response
```

The response includes:

```json
{
  "session_id": "returned-session-id",
  "answer": "The answer generated from the uploaded documents.",
  "sources": [
    {
      "filename": "yourfile.pdf",
      "chunk_index": 0,
      "score": 0.81
    }
  ],
  "booking_status": "none"
}
```

Save `session_id` and send it with later messages to preserve conversation
history:

```powershell
$body = @{
  session_id = "returned-session-id"
  message = "Can you explain that in one sentence?"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/v1/chat `
  -ContentType "application/json" `
  -Body $body
```

If no relevant document chunk is found, the API returns a fixed
`I couldn't find that in the uploaded documents.` response instead of asking
the LLM to guess.

Clear a chat session and its open booking state:

```powershell
Invoke-WebRequest `
  -Method Delete `
  -Uri "http://localhost:8000/api/v1/chat/<session-id>"
```

### Book an interview

Booking is handled through the same chat endpoint. The application asks for
missing details and then displays a summary for confirmation.

Example conversation:

```text
User: I want to book an interview
Assistant: What's your full name?

User: My name is Ram Sharma, ram@example.com
Assistant: Which date would you like for the interview?

User: Next Tuesday at 2pm
Assistant: [booking summary] Shall I confirm it? (yes/no)

User: yes
Assistant: Your interview is booked ...
```

The first response returns a `session_id`. Send that same ID with every later
message in the booking conversation.

Possible `booking_status` values are:

| Status | Meaning |
|---|---|
| `none` | No booking is currently active |
| `collecting` | The application is collecting or correcting details |
| `confirming` | All details are present and confirmation is required |
| `confirmed` | The booking was saved |
| `cancelled` | The booking was cancelled |

A document question asked during an open booking is answered normally, with a
reminder about the unfinished booking. Booking replies are written by the
application code; the LLM only extracts possible fields from the user's
message. This keeps confirmation and cancellation behavior predictable.

List recent bookings:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/bookings
```

This endpoint currently has no authentication and includes email addresses.
Do not expose it publicly without adding access control.

## Configuration

Copy `.env.example` to `.env` and adjust values only when needed.

| Variable | Purpose | Default |
|---|---|---|
| `QDRANT_URL` | Qdrant connection URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Optional Qdrant API key | empty |
| `QDRANT_COLLECTION` | Vector collection name | `documents` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `DATABASE_URL` | Async PostgreSQL URL | local `rag` database |
| `GROQ_API_KEY` | Required Groq secret | none |
| `GROQ_MODEL` | Groq chat model | `openai/gpt-oss-120b` |
| `EMBEDDING_MODEL` | FastEmbed model | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DIM` | Vector size | `384` |
| `CHUNK_SIZE` | Default chunk size | `800` |
| `CHUNK_OVERLAP` | Default overlap | `100` |
| `RETRIEVAL_TOP_K` | Maximum retrieved chunks | `5` |
| `RETRIEVAL_MIN_SCORE` | Minimum Qdrant score | `0.25` |
| `RETRIEVAL_SCORE_MARGIN` | Allowed distance from best score | `0.2` |
| `CHAT_HISTORY_LENGTH` | Saved chat turns per session | `10` |
| `CHAT_TTL_SECONDS` | Chat memory lifetime | `86400` |
| `BOOKING_TTL_SECONDS` | Temporary booking lifetime | `1800` |

When running in Docker, Compose replaces `localhost` service URLs with
`postgres`, `redis`, and `qdrant` automatically.

## Run locally without the API container

Docker is still required for PostgreSQL, Redis, and Qdrant.

Create a virtual environment and install development dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Make sure `.env` contains URLs for services running on the host:

```dotenv
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/rag
GROQ_API_KEY=your_groq_api_key_here
```

Start only the backing services:

```powershell
docker compose up -d postgres redis qdrant
```

Start FastAPI from the repository root:

```powershell
uvicorn app.main:app --reload
```

The local API is then available at
[http://localhost:8000/docs](http://localhost:8000/docs).

## Tests and code quality

Install development dependencies first:

```powershell
pip install -r requirements-dev.txt
```

Run the automated test suite:

```powershell
pytest
```

The default test command excludes tests marked `live`. The tests use fakes and
local in-memory or temporary backends, so they do not require a Groq key or a
running Docker stack.

Run the real Groq smoke tests intentionally:

```powershell
pytest -m live
```

These tests are skipped when `GROQ_API_KEY` is missing. They check model-based
field extraction, relative-date handling, document answering, and an HTTP
booking conversation. They use fake Qdrant, Redis, SQL, and embedding
components; only the Groq model call is real.

Run formatting, linting, and type checking:

```powershell
ruff check .
ruff format --check .
mypy app
```

## Project structure

```text
app/
  main.py                 Application creation and startup/shutdown wiring
  api/
    documents.py          Upload, list, and delete document endpoints
    chat.py               Conversational RAG and chat-clear endpoints
    bookings.py           Recent bookings endpoint
    health.py             Dependency health endpoint
    deps.py               FastAPI dependency wiring
    errors.py             Consistent exception-to-HTTP mapping
  core/
    config.py             Environment settings and validation
    database.py           Async SQLAlchemy engine setup
  models/
    db.py                 Document and Booking database models
  services/
    extractor.py          PDF and text extraction
    chunker.py            Fixed and sentence-aware chunking
    embedder.py           FastEmbed embedding adapter
    vector_store.py       Qdrant collection and vector operations
    ingest.py             Upload, deduplication, and cleanup workflow
    rag.py                Retrieval, score filtering, and answer generation
    query_rewriter.py     Follow-up question rewriting
    chat.py               Chat orchestration
    chat_memory.py        Redis chat history
    llm.py                Groq client adapter
    booking.py             Booking validation rules
    booking_extractor.py  Structured LLM field extraction
    booking_flow.py       Booking state machine
    booking_state.py      Temporary Redis booking state
    booking_repository.py PostgreSQL booking persistence
    health.py              PostgreSQL, Redis, and Qdrant probes
tests/
  test_*.py               Unit and integration-style tests with fakes
  live/test_smoke.py      Opt-in tests using the real Groq API
docker-compose.yml        API and backing-service orchestration
dockerfile                Production-style API image
requirements.txt          Runtime dependencies
requirements-dev.txt      Runtime plus development/test tools
pyproject.toml            Pytest, Ruff, and mypy configuration
.env.example              Safe configuration template
```

The `try_*.py` files and `yourfile.pdf` are local scratch/sample files and
should be removed before publishing a clean repository.

## Important design decisions

### File deduplication

The application hashes the complete uploaded file with SHA-256. A repeated
upload returns the existing document instead of embedding and storing duplicate
chunks. A database uniqueness constraint also protects against simultaneous
duplicate uploads.

### Two chunking strategies

- `fixed` creates overlapping character windows.
- `sentence` keeps sentences together when possible and still splits very
  long sentences so no chunk exceeds the configured size.

### Retrieval safety

Retrieval uses a minimum similarity score and a score margin from the best
match. Weak, unrelated chunks are discarded. Document text is placed inside
data tags and closing tags are removed before it is sent to the LLM, reducing
the risk of document prompt injection.

### Booking safety

The LLM extracts structured booking fields, but application code validates
dates, times, email addresses, confirmation, cancellation, and duplicate
bookings. A bare `yes`, `no`, or `okay` in a new session cannot start a booking
by itself.

## Data and reset behavior

Docker uses named volumes named `pgdata` and `qdrantdata`. Redis data is stored
according to the Redis container configuration.

Stop containers without deleting data:

```powershell
docker compose down
```

Stop containers and delete the database/vector volumes:

```powershell
docker compose down -v
```

Use `down -v` when intentionally resetting the local database, or after a
database schema change. There are currently no migration files.

## Known limitations

- There is no authentication.
- `GET /api/v1/bookings` exposes stored email addresses.
- Booking uniqueness is currently per session, so two different sessions can
  select the same time slot.
- A document row can remain in `processing` after an unexpected process crash;
  a startup cleanup sweep is not implemented.
- The first image build downloads roughly 90 MB of embedding model data.
- The configured Groq reasoning model can respond more slowly than a
  conventional chat model.
- Retrieval thresholds were tuned against a small document set and may need
  adjustment for a larger production corpus.
- PDF extraction handles text PDFs, not scanned PDFs that require OCR.
- There are no database migrations; schema changes require recreating local
  volumes.

## Security checklist before publishing

```powershell
git status --short
git ls-files .env .env.example
git log -p | Select-String -Pattern "gsk_|password|secret"
```

The output should show `.env.example`, but not `.env`. If a real API key was
ever committed, revoke and rotate it immediately. Removing it in a later
commit is not enough.
