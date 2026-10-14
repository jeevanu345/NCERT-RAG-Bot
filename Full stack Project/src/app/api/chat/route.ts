import { proxy } from "@/lib/backend";
export const runtime = "nodejs";
export const maxDuration = 120;
export async function POST(request: Request) {
  return proxy(request, "/api/chat");
}
