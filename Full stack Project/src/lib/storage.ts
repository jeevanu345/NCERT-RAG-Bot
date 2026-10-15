import type { Conversation, Message } from "./types";

const STORAGE_KEY = "ncert-sessions-v1";
// Read only this app's versioned schema; malformed browser data never becomes chat input.
export function loadConversations(scope = "guest"): Conversation[] {
  try {
    const parsed: unknown = JSON.parse(
      localStorage.getItem(
        scope === "guest" ? STORAGE_KEY : `${STORAGE_KEY}:${scope}`,
      ) ?? "[]",
    );
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (c): c is Conversation =>
          c &&
          typeof c.id === "string" &&
          typeof c.title === "string" &&
          Array.isArray(c.messages) &&
          c.messages.every(
            (m: Message) =>
              m &&
              ["user", "assistant"].includes(m.role) &&
              typeof m.content === "string" &&
              m.content.length <= 20000 &&
              (m.saved === undefined || typeof m.saved === "boolean") &&
              (m.cards === undefined ||
                (Array.isArray(m.cards) &&
                  m.cards.length <= 10 &&
                  m.cards.every(
                    (card) =>
                      card &&
                      typeof card.question === "string" &&
                      typeof card.answer === "string",
                  ))) &&
              (m.sources === undefined ||
                (Array.isArray(m.sources) &&
                  m.sources.every(
                    (s) =>
                      s &&
                      typeof s.citation === "number" &&
                      typeof s.document_id === "string" &&
                      typeof s.name === "string" &&
                      typeof s.page === "number" &&
                      typeof s.excerpt === "string",
                  ))),
          ),
      )
      .slice(0, 20)
      .map((c) => ({ ...c, messages: c.messages.slice(-40) }));
  } catch {
    return [];
  }
}
export function saveConversations(
  conversations: Conversation[],
  scope = "guest",
) {
  localStorage.setItem(
    scope === "guest" ? STORAGE_KEY : `${STORAGE_KEY}:${scope}`,
    JSON.stringify(conversations.filter((c) => c.messages.length).slice(0, 20)),
  );
}
export function downloadConversation(conversation: Conversation) {
  const content =
    `# ${conversation.title}\n\n` +
    conversation.messages
      .map(
        (message) =>
          `## ${message.role === "user" ? "You" : "NCERT Tutor"}\n\n${message.content}\n` +
          (message.sources?.length
            ? "\nSources:\n" +
              message.sources
                .map((s) => `- [${s.citation}] ${s.name}, PDF page ${s.page}`)
                .join("\n") +
              "\n"
            : ""),
      )
      .join("\n");
  const url = URL.createObjectURL(
    new Blob([content], { type: "text/markdown;charset=utf-8" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${conversation.title.replace(/[^\p{L}\p{N} -]/gu, "").slice(0, 70) || "study-session"}.md`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
