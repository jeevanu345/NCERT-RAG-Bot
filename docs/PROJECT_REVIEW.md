# Project review and upgrade

## Repository findings

The repository contained three independent implementations:

1. The root Streamlit prototype used LangChain, Chroma, local embeddings, and a local chat model. Its cached conversational chain could share mutable memory across sessions. It lost source page information during extraction and depended on a separate model daemon.
2. `Full stack Project/` contained a Next.js interface and FastAPI backend. It required multiple cloud providers, returned incomplete upload errors, lacked working settings, and showed a placeholder source rather than a real library. Chat and library state were separate. Duplicate Next.js configuration files and an incorrect root `lib/` ignore pattern also hid necessary frontend helpers.
3. `From this /` is an unrelated ERP/CRM template, not an NCERT application. It is excluded from the canonical setup and was left intact.

The upgrade consolidates study behavior around the second implementation and provides root-level commands so the starting point is unambiguous.

## Changes made

| Area          | Result                                                                                                                                                    |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UI            | Reference-inspired vector book logo, animated interactive landing page, responsive study workspace, dark mode, six palettes, and configurable local fonts |
| Accounts      | Google, Microsoft, and Apple OAuth code; encrypted sessions; isolated account/guest libraries; custom sign-in, sign-out, and error pages                  |
| Setup         | One setup command and one development command for both services                                                                                           |
| Models        | NVIDIA NIM BYOK, optional server key, selectable model, real connection test                                                                              |
| Credentials   | Memory by default, optional tab storage, no database persistence or provider error leakage                                                                |
| Retrieval     | Page-aware chunks, persistent SQLite FTS5/BM25 search, verified account/guest workspace scope                                                             |
| Uploads       | Real extraction/indexing status, duplicate detection, text/size/page limits, batch upload, scanned/encrypted/corrupt PDF errors                           |
| Library       | Search, textbook selection, scope controls, and removal with index cleanup                                                                                |
| Answers       | Retrieved source excerpts, PDF page references, conversation follow-ups, grounded no-match response                                                       |
| Revision      | Explain, revise, practice, and validated interactive flashcards                                                                                           |
| Sessions      | Browser history, search, saved answers, Markdown export, stop and retry                                                                                   |
| Accessibility | Keyboard navigation, labeled controls, focus-trapped dialogs, skip link, mobile drawer, reduced motion                                                    |
| Verification  | API unit/integration tests, browser flow tests, TypeScript, ESLint, production build, CI                                                                  |
| Maintenance   | Shared chunking, smaller dependency set, updated framework packages, removed local model configuration                                                    |

## Design decisions

The reference search included light sidebars and conversation lists from the 21st catalog. The implementation reuses the installed Button/Input primitives and Radix dialogs, rather than importing a second UI kit. A light or dark canvas, six accent palettes, and selectable body/display fonts keep the textbook and questions central. Fonts are self-hosted from pinned npm packages. The vector book/person logo follows the supplied reference; green remains its brand color while interface accents can change. The landing page uses interactive examples and restrained entrance/scroll animations, with reduced-motion support.

The revision and flashcard features also draw on the source-based study patterns described in [Google’s NotebookLM student feature overview](https://blog.google/innovation-and-ai/models-and-research/google-labs/notebooklm-student-features/).

The interface groups actions around the student’s intent: study, library, and saved answers. Starter prompts fill the composer for review before sending. Connection and upload steps remain accessible from the welcome state. Errors preserve the question and offer recovery.

## Remaining limits and next improvements

- **Semantic retrieval evaluation:** add a small, versioned, redistributable benchmark and measure recall, groundedness, and citation accuracy before claiming improvements. Hosted NVIDIA embeddings/reranking could use the same key, but add API calls and need their own evaluation.
- **OCR and formula support:** handle scanned material and page images deliberately; this version clearly rejects scans without readable text.
- **Public multi-user hosting:** configure the implemented OAuth providers, set required authentication, and add hosting quotas, request controls, backups, and persistent hosted storage before public deployment. OAuth credentials and a live consent flow must be verified in the owner’s registered developer accounts.
- **Streaming:** answers arrive as one completion. Stop cancels the browser/proxy wait; it cannot guarantee that a provider request already in flight stops running.
- **Storage:** local sessions are bounded and browser-specific. Export important conversations. Browser workspace cookies are separate from chat history.
- **Accuracy:** source markers and flashcard structure are validated as software data. They do not prove the model’s factual claims. No measured accuracy claim is made.
- **Dependency advisories:** compatible updates were applied and an unused CLI dependency removed. The current framework ESLint chain still includes an unpatched `braces` advisory through `fast-glob`/`micromatch`. It is a development-time lint dependency. Do not force npm’s suggested downgrade of Next.js’s ESLint configuration to an unrelated major release; track the upstream fix.

The included original sample notes and tests do not redistribute NCERT textbook content. Upload your own authorized copy for actual study.
