import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { randomUUID } from "node:crypto";
import { currentAccount } from "@/auth";
import { authenticationRequired } from "@/lib/auth-config";

const BASE = (
  process.env.PYTHON_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");
const COOKIE = "ncert-workspace";
const MAX_BYTES = 20 * 1024 * 1024;

export async function proxy(
  request: Request,
  path: string,
  method = request.method,
) {
  // Browser writes must originate from this application.
  const origin = request.headers.get("origin");
  // Next.js can normalize request.url to localhost even when the browser uses
  // 127.0.0.1. The Host header retains the browser-facing authority.
  const requestURL = new URL(request.url);
  const expectedOrigin = process.env.AUTH_URL
    ? new URL(process.env.AUTH_URL).origin
    : `${requestURL.protocol}//${request.headers.get("host") ?? requestURL.host}`;
  if (method !== "GET" && origin && origin !== expectedOrigin) {
    return NextResponse.json(
      { error: "This request must come from NCERT Tutor." },
      { status: 403 },
    );
  }
  const account = await currentAccount();
  const expectedIdentity = request.headers.get("x-ncert-identity");
  if (expectedIdentity && expectedIdentity !== (account?.id || "guest"))
    return NextResponse.json(
      { error: "Your account changed. Reload this page before continuing." },
      { status: 409 },
    );
  if (!account && authenticationRequired() && path !== "/api/health")
    return NextResponse.json(
      { error: "Please sign in to use your study workspace." },
      { status: 401 },
    );
  const jar = await cookies();
  const existing = jar.get(COOKIE)?.value;
  const id =
    account?.id ||
    (existing &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
      existing,
    )
      ? existing
      : randomUUID());
  const headers: Record<string, string> = {
    "X-Workspace-ID": id,
    "X-Workspace-Kind": account ? "account" : "guest",
  };
  const auth = request.headers.get("authorization");
  if (auth) headers.Authorization = auth;
  if (request.headers.get("content-type"))
    headers["Content-Type"] = request.headers.get("content-type")!;
  try {
    let body: Uint8Array | undefined;
    if (method !== "GET" && method !== "DELETE") {
      const declared = Number(request.headers.get("content-length") ?? 0);
      if (declared > MAX_BYTES)
        return NextResponse.json(
          { error: "PDFs must be 20 MB or smaller." },
          { status: 413 },
        );
      const reader = request.body?.getReader();
      const chunks: Uint8Array[] = [];
      let size = 0;
      if (reader) {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          size += value.length;
          if (size > MAX_BYTES) {
            await reader.cancel();
            return NextResponse.json(
              { error: "Request is too large. PDFs must be 20 MB or smaller." },
              { status: 413 },
            );
          }
          chunks.push(value);
        }
      }
      body = new Uint8Array(size);
      let offset = 0;
      for (const chunk of chunks) {
        body.set(chunk, offset);
        offset += chunk.length;
      }
    }
    const upstream = await fetch(`${BASE}${path}`, {
      method,
      headers,
      body: body as BodyInit | undefined,
      cache: "no-store",
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(115000)]),
    });
    const data = await upstream.json().catch(() => null);
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      return NextResponse.json(
        {
          error:
            "The study service returned an unreadable response. Please retry.",
        },
        { status: 502, headers: { "Cache-Control": "no-store" } },
      );
    }
    // Pydantic validation details are arrays, not user-facing text.
    if (Array.isArray(data.detail))
      data.detail =
        "Check your question, model ID, and selected textbooks, then try again.";
    const response = NextResponse.json(data, {
      status: upstream.status,
      headers: { "Cache-Control": "no-store" },
    });
    if (!account && (!existing || existing !== id))
      response.cookies.set(COOKIE, id, {
        httpOnly: true,
        sameSite: "strict",
        secure: new URL(expectedOrigin).protocol === "https:",
        path: "/",
        maxAge: 60 * 60 * 24 * 30,
      });
    return response;
  } catch (error) {
    const timeout =
      error instanceof Error &&
      (error.name === "TimeoutError" || error.name === "AbortError");
    return NextResponse.json(
      {
        error: timeout
          ? "The study service took too long to respond. Please retry."
          : "The study service is offline. Start the Python API and try again.",
      },
      { status: timeout ? 504 : 503 },
    );
  }
}
