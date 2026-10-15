# NCERT Tutor architecture

The canonical app lives in `Full stack Project/`. Root npm scripts orchestrate setup, development, validation, and a local production run.

```text
Browser: landing page, study workspace, library, saved answers
  | same-origin /api/* requests; optional NVIDIA key in Authorization
  v
Next.js route handlers (Node.js)
  | Verified OAuth session or guest cookie -> trusted workspace ID + kind
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

- `src/app/page.tsx` and `components/LandingExperience.tsx`: public landing page and interactive sample previews.
- `src/app/app/page.tsx` and `components/StudyWorkspace.tsx`: active session, library scope, composer, upload/error state, account-scoped browser persistence.
- `src/auth.ts` and `src/app/api/auth/[...nextauth]`: Google, Microsoft Entra ID, and Apple OAuth; encrypted JWT sessions; provider-based account identity.
- `src/lib/auth-config.ts`: safe return destinations, provider readiness, and optional required-auth policy.
- `src/components/AppearanceProvider.tsx`, `src/lib/appearance.ts`, and `src/app/appearance.css`: validated persisted appearance, system/light/dark modes, six accent palettes, local variable fonts.
- `src/components/`: navigation, library, connection settings, source-backed answers, flashcard review, and accessible dialogs.
- `src/lib/backend.ts`: browser-to-Python proxy. Never accepts a browser-supplied workspace header.
- `src/lib/storage.ts`: versioned browser history and Markdown export; no key storage.
- `api/index.py`: validated endpoints and bounded PDF processing.
- `api/rag_pipeline.py`: SQLite lifecycle, indexing/search, provider errors, prompt construction, and flashcard validation.
- `api/chunking.py`: dependency-free chunking with page boundaries preserved by the caller.

## API surface

| Method | Endpoint                 | Purpose                                                                          |
| ------ | ------------------------ | -------------------------------------------------------------------------------- |
| GET    | `/api/health`            | Provider name, default model, whether a server key exists; never returns the key |
| POST   | `/api/connection`        | Tiny real completion to verify the supplied key/model                            |
| GET    | `/api/documents`         | Documents in the verified account or guest workspace                             |
| POST   | `/api/upload?filename=…` | Raw PDF bytes, extraction, and atomic index insertion                            |
| DELETE | `/api/documents/{id}`    | Scoped deletion, cascading passage/FTS cleanup                                   |
| POST   | `/api/chat`              | Messages, mode, model ID, optional document IDs; returns answer/sources/cards    |

## Data lifecycle

SQLite stores document metadata and extracted text. A SHA-256 digest prevents duplicate imports within a workspace. A transaction serializes quota checks and indexing. Page numbers refer to physical PDF pages starting at 1, which can differ from printed textbook page labels. The original upload is not retained.

Search always filters by workspace and optional document selection. Broad prompts sample passages across the selected content; specific questions use ranked lexical matches. A follow-up without a match retries retrieval with the last two user questions. Without relevant material, the API returns a clarification instead of calling the model without sources.

The browser owns chat history and sends the last 12 messages. The server has no shared mutable chat memory. PDF content is treated as untrusted data in the model prompt. Flashcard JSON and source marker ranges are validated before becoming an interactive review.

## Hosting boundary

The default runs locally with optional guest study. Next.js maps a verified OAuth provider/subject to a stable account UUID and forwards it to a separate `account:` library namespace. A random, HTTP-only, same-site cookie identifies guest libraries; guest cookies and browser-supplied headers cannot select account data. The proxy checks optional client identity markers to reject requests from stale account screens.

Auth.js handles provider validation and encrypted HTTP-only sessions lasting up to seven days. Provider access/refresh tokens are not retained in browser session data. Account chat histories use separate browser-storage keys. Cookie changes re-render the account UI, so sign-out performs the selected local cleanup before the server redirects. Cross-tab identity signals clear remembered keys and refresh stale screens.

Set `AUTH_REQUIRED=true` for authenticated-only study routes; missing credentials fail closed. Canonical `AUTH_URL` origin checks support a trusted HTTPS reverse proxy. FastAPI trusts Next.js and must remain private. Public hosting also needs usage limits, backups, and persistent storage. See [OAuth setup](docs/OAUTH_SETUP.md). Do not deploy SQLite to an ephemeral function filesystem and expect durable documents.

No deployment has been performed as part of this source-code upgrade.
