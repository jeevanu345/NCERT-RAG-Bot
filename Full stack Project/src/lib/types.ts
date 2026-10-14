export type Textbook = {
  id: string;
  name: string;
  pages: number;
  chunks: number;
};
export type Source = {
  citation: number;
  document_id: string;
  name: string;
  page: number;
  excerpt: string;
  cited: boolean;
};
export type Flashcard = { question: string; answer: string };
export type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  saved?: boolean;
  cards?: Flashcard[];
};
export type Conversation = { id: string; title: string; messages: Message[] };
export type StudyMode = "explain" | "revise" | "practice" | "flashcards";
export const DEFAULT_MODEL = "meta/llama-3.3-70b-instruct";
export async function readResponse(response: Response) {
  const data = await response.json().catch(() => null);
  if (!data || typeof data !== "object" || Array.isArray(data))
    throw new Error(
      "The app received an unreadable response. Please try again.",
    );
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : (data.error ?? "Something went wrong. Please try again."),
    );
  return data;
}
