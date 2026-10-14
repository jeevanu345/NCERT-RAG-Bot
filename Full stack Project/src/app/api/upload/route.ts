import { proxy } from "@/lib/backend";
export const runtime = "nodejs";
export const maxDuration = 120;
export async function POST(request: Request) {
  const filename = new URL(request.url).searchParams.get("filename");
  if (!filename)
    return Response.json({ error: "Choose a PDF to upload." }, { status: 400 });
  return proxy(request, `/api/upload?filename=${encodeURIComponent(filename)}`);
}
