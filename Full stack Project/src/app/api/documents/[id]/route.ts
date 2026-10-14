import { proxy } from "@/lib/backend";
export async function DELETE(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  return proxy(request, `/api/documents/${encodeURIComponent(id)}`);
}
