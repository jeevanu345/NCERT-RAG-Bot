import { handlers } from "@/auth";
import { authConfigured, authenticationRequired } from "@/lib/auth-config";
import type { NextRequest } from "next/server";
export const runtime = "nodejs";
function unavailable() {
  return Response.json(
    {
      error: authenticationRequired()
        ? "Sign-in is not configured yet. Please contact the app owner."
        : "Sign-in is not configured yet. You can use the guest workspace.",
    },
    { status: 503, headers: { "Cache-Control": "no-store" } },
  );
}
export async function GET(request: NextRequest) {
  return authConfigured() ? handlers.GET(request) : unavailable();
}
export async function POST(request: NextRequest) {
  return authConfigured() ? handlers.POST(request) : unavailable();
}
