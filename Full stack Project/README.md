This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

Install JavaScript dependencies:

```bash
npm install
```

Create and activate a Python virtual environment, then install backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt uvicorn
```

Run the FastAPI backend (required for chat):

```bash
python3 -m uvicorn api.index:app --reload --port 8000
```

In another terminal, run the Next.js development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

If port 3000 is occupied, Next.js will use another port (for example, 3001). This project now derives the active origin automatically for upload processing.

The chat route is handled at `src/app/api/chat/route.ts` and forwards requests to FastAPI at `http://127.0.0.1:8000` by default. Override with:

```bash
PYTHON_API_BASE_URL=http://127.0.0.1:8000
```

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
