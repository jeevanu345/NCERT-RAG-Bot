# NCERT Tutor web app

This is the canonical Next.js + FastAPI implementation. See the [root README](../README.md) for setup, NVIDIA NIM BYOK, features, privacy, and limits.

From the repository root:

```bash
npm run setup
npm run dev
```

From this directory, after root setup:

```bash
npm run dev:api       # terminal 1
npm run dev           # terminal 2
npm run test:api
npm run lint
npm run typecheck
npm run build
npm run test:e2e
```

The Python environment lives at `../.venv`. Copy `.env.example` to `.env.local` only if you want optional server-side configuration. Browser BYOK works without it.
