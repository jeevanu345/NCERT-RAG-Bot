import { NextResponse } from 'next/server';

type ChatMessage = {
  role: string;
  content: string;
};

type ChatRequest = {
  messages?: ChatMessage[];
};

const PYTHON_API_BASE_URL = process.env.PYTHON_API_BASE_URL ?? 'http://127.0.0.1:8000';

export const runtime = 'nodejs';

export async function POST(request: Request) {
  let payload: ChatRequest;

  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  if (!Array.isArray(payload.messages) || payload.messages.length === 0) {
    return NextResponse.json({ error: 'Messages array cannot be empty' }, { status: 400 });
  }

  try {
    const upstreamResponse = await fetch(`${PYTHON_API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: payload.messages }),
    });

    const upstreamJson = await upstreamResponse.json().catch(() => null);

    if (!upstreamResponse.ok) {
      const detail =
        upstreamJson?.detail ?? upstreamJson?.error ?? 'Failed to generate a response from the backend';
      return NextResponse.json({ error: 'Chat backend error', detail }, { status: upstreamResponse.status });
    }

    return NextResponse.json(upstreamJson ?? {});
  } catch {
    return NextResponse.json(
      {
        error: 'Chat backend unavailable',
        detail:
          'Start the FastAPI server first: python3 -m uvicorn api.index:app --reload --port 8000',
      },
      { status: 503 }
    );
  }
}
