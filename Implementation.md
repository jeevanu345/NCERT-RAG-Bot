# NCERT Tutor architecture

The canonical app lives in `Full stack Project/`. Root npm scripts orchestrate setup, development, validation, and a local production run.

```text
Browser: study workspace, library, saved answers
  | same-origin /api/* requests; optional NVIDIA key in Authorization
  v
Next.js route handlers (Node.js)
  | HTTP-only workspace cookie -> trusted X-Workspace-ID header
  | body-size limits, origin checks, no-store responses, request timeout
  v
FastAPI (127.0.0.1:8000)
  |-- PDF extraction with pypdf -> overlapping page-preserving chunks
  |-- workspace-scoped documents/passages -> SQLite FTS5/BM25
  |-- grounded system prompt + retrieved passages + recent conversation
  v
NVIDIA hosted chat completions API
  |
  v
Answer + retrieved passages + citation markers + optional validated flashcards
```

## Ownership

- `src/app/page.tsx`: active session, library scope, composer, upload/error state, browser persistence.
- `src/components/`: navigation, library, connection settings, source-backed answers, flashcard review, and accessible dialogs.
- `src/lib/backend.ts`: browser-to-Python proxy. Never accepts a browser-supplied workspace header.
- `src/lib/storage.ts`: versioned browser history and Markdown export; no key storage.
- `api/index.py`: validated endpoints and bounded PDF processing.
- `api/rag_pipeline.py`: SQLite lifecycle, indexing/search, provider errors, prompt construction, and flashcard validation.
- `api/chunking.py`: dependency-free chunking with page boundaries preserved by the caller.

## API surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Provider name, default model, whether a server key exists; never returns the key |
| POST | `/api/connection` | Tiny real completion to verify the supplied key/model |
| GET | `/api/documents` | Documents in the current browser workspace |
| POST | `/api/upload?filename=…` | Raw PDF bytes, extraction, and atomic index insertion |
| DELETE | `/api/documents/{id}` | Scoped deletion, cascading passage/FTS cleanup |
| POST | `/api/chat` | Messages, mode, model ID, optional document IDs; returns answer/sources/cards |

## Data lifecycle

SQLite stores document metadata and extracted text. A SHA-256 digest prevents duplicate imports within a workspace. A transaction serializes quota checks and indexing. Page numbers refer to physical PDF pages starting at 1, which can differ from printed textbook page labels. The original upload is not retained.

Search always filters by workspace and optional document selection. Broad prompts sample passages across the selected content; specific questions use ranked lexical matches. A follow-up without a match retries retrieval with the last two user questions. Without relevant material, the API returns a clarification instead of calling the model without sources.

The browser owns chat history and sends the last 12 messages. The server has no shared mutable chat memory. PDF content is treated as untrusted data in the model prompt. Flashcard JSON and source marker ranges are validated before becoming an interactive review.

## Hosting boundary

The default is a single-user local application, not a public service. Next.js creates a random, HTTP-only, same-site workspace cookie. FastAPI trusts Next.js and must remain private. To host publicly, replace this boundary with authenticated identities and add usage limits and persistent storage. Do not deploy SQLite to an ephemeral function filesystem and expect durable documents.

No deployment has been performed as part of this source-code upgrade.
