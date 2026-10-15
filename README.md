# NCERT Tutor

A simple study workspace for Class 10 NCERT textbooks. Bring your **NVIDIA NIM API key**, upload a PDF, and ask questions with textbook page references.

The main application is **Next.js + FastAPI** in `Full stack Project/`. All model generation uses NVIDIA’s hosted API. No local model installation, GPU, OpenAI account, Pinecone account, or embedding download is required.

## Run locally

Requirements: Node.js 20.9+ and Python 3.11–3.14 with SQLite FTS5 (included in standard Python installations).

```bash
npm run setup
npm run dev
```

Open **http://127.0.0.1:3000** for the landing page; the study workspace is at **/app**. The development command starts both the UI and Python service, and stops both with Ctrl+C.

1. Open **Settings** or **Connect NVIDIA**.
2. Get a developer key from the [NVIDIA API Catalog](https://build.nvidia.com/meta/llama-3_3-70b-instruct), paste it, and select **Test & save connection**.
3. Add a PDF in **Textbook library**, or try the included original sample science notes.
4. Select a textbook and study mode, then ask a question.

NVIDIA describes free developer API access for prototyping in its [FAQ](https://docs.api.nvidia.com/nim/docs/product). Model access, rate limits, and allowances depend on your account. The app surfaces authentication, quota, model-access, and timeout errors; it does not silently switch providers.

## Study features

- **Explain:** simple explanations and worked examples.
- **Revise:** concise revision notes and key definitions.
- **Practice:** exam-style questions with answers.
- **Flashcards:** reveal answers, move between cards, mark remembered cards, and restart a review.
- Expandable source passages with **PDF page numbers** and citation markers.
- A searchable textbook library, multiple PDF uploads, duplicate detection, and document removal.
- Questions scoped to one textbook or the whole library.
- Browser-saved study sessions, session search, saved answers, copy, and Markdown export.
- Retry failed questions and stop waiting for an answer.
- Light, dark, and system modes; six palettes, four body fonts, and three display fonts in **Settings → Appearance**.
- A responsive animated landing page with interactive study examples.
- Google, Microsoft, and Apple sign-in, account menus, and sign-out with optional chat-cache removal.
- Keyboard shortcuts, accessible dialogs, mobile navigation, and reduced-motion support.

Press **⌘/Ctrl K** for a new session; **Enter** sends, and **Shift + Enter** adds a line. Chat requests include up to the last 12 messages. The browser retains up to 20 sessions and the latest 40 messages per session. Export longer sessions to keep a copy. Saved answers belong to their sessions.

## API key and privacy

Keys are held in page memory by default. **Remember my key for this tab** opts into `sessionStorage`; the key must be tested again after reloading. Disconnect removes it. Keys are never stored in the library database or chat history.

Alternatively, copy the example configuration:

```bash
cp "Full stack Project/.env.example" "Full stack Project/.env.local"
```

Set `NVIDIA_API_KEY` there and restart the Python API. This is optional; browser BYOK needs no environment file. An explicitly supplied browser key takes precedence over a server key.

The default model is `meta/llama-3.3-70b-instruct`. Settings also provides `meta/llama-3.1-8b-instruct` and accepts a custom NVIDIA catalog model ID. All requests use `https://integrate.api.nvidia.com/v1/chat/completions`.

Text is extracted and indexed locally in `Full stack Project/.data/library.sqlite3`. The original uploaded PDF bytes are discarded after extraction. Questions, recent conversation context, and selected relevant excerpts are sent to NVIDIA. Document and chat deletion are available in the UI. Guest sessions identify libraries through an HTTP-only cookie lasting 30 days; clearing or expiring it creates a new guest library. Signed-in accounts have a separate library based on their verified provider identity. Chat history stays in that browser’s local storage, scoped to the account. See **/privacy** for the data lifecycle.

## Optional Google, Microsoft, and Apple accounts

Follow [the OAuth setup guide](docs/OAUTH_SETUP.md) for provider registrations, environment variables, exact callback URLs, and Apple client-secret generation. Run `npm run auth:secret` to generate the private session secret without printing it. Provider buttons become available after their server credentials are configured; Apple requires a registered HTTPS origin. Live OAuth credentials are not bundled with the project.

Guest mode is available by default. Set `AUTH_REQUIRED=true` to require sign-in for the study workspace and API. Guest books and chats remain separate from signed-in data; chat history is stored on the device rather than synced. Open the same origin specified by `AUTH_URL` when OAuth is configured.

## Commands

```bash
npm test                  # 23 backend tests; provider calls are mocked
npm run lint
npm run typecheck
npm run build
npm run start             # UI + API with the production build, bound to localhost
npm run test:e2e           # 23 browser/configuration checks; run build first
```

Install the test browser once with `cd "Full stack Project" && npx playwright install chromium`.

To use different local ports:

```bash
PORT=3001 API_PORT=8001 npm run dev
```

To run services separately:

```bash
cd "Full stack Project"
npm run dev:api            # terminal 1, localhost:8000
npm run dev               # terminal 2, localhost:3000
```

An optional terminal client uses the same pipeline:

```bash
NVIDIA_API_KEY=your_key .venv/bin/python rag_pipeline.py --pdf /path/to/chapter.pdf
```

Prefer the browser key field or `.env.local` to putting a real key in shell history. `streamlit_app.py` now forwards to the web application; `chunking.py` re-exports the shared chunker. The unrelated ERP/CRM template under `From this /` is not part of this app or its install process.

## How it works and limits

PDF → page-preserving text chunks → SQLite FTS5/BM25 search → relevant excerpts and conversation → NVIDIA NIM → answer and page references.

Retrieval is lexical and uses English stemming; it does not download or run a model. Questions with wording similar to the textbook work best. Broad revision prompts sample the selected material, so they are not exhaustive textbook summaries. Cross-language retrieval and mathematical notation can be weaker than semantic search.

Use text-based PDFs up to **20 MB**, **700 pages**, and **20 documents per library**. Scans, handwriting, diagrams, and image-only formulas need OCR before upload. The included sample is original demonstration material, not an NCERT textbook.

This setup is for local development and personal study. It intentionally binds both services to localhost. The Python service trusts the Next.js proxy’s workspace header: do not expose it directly. For public deployment, configure OAuth, require authentication, and add hosting-level usage controls, backups, and persistent storage. SQLite data will not survive an ephemeral serverless deployment without a persistent backend.

Generated answers and citations can be wrong; check them against the original textbook. Automated tests verify software behavior using original fixtures and mocked provider responses, not educational accuracy. A live model check needs your key in Settings.

See [the project review](docs/PROJECT_REVIEW.md) and [architecture notes](Implementation.md) for what changed and sensible next improvements.
