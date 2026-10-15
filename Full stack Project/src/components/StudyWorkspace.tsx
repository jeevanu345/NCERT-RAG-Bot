"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { Dialog } from "radix-ui";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  BookOpen,
  Bookmark,
  Check,
  ChevronRight,
  Download,
  Layers,
  Lightbulb,
  ListChecks,
  LoaderCircle,
  Menu,
  Paperclip,
  RotateCcw,
  Square,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { LogoMark } from "@/components/brand/Logo";
import AccountMenu from "@/components/AccountMenu";
import ThemeToggle from "@/components/ThemeToggle";
import type { AccountInfo } from "@/lib/auth-config";
import Sidebar, { type View } from "@/components/Sidebar";
import Library from "@/components/Library";
import Modal from "@/components/Modal";
import type { Connection } from "@/components/ConnectionSettings";
import {
  DEFAULT_MODEL,
  readResponse,
  type Conversation,
  type Message,
  type StudyMode,
  type Textbook,
} from "@/lib/types";
import {
  downloadConversation,
  loadConversations,
  saveConversations,
} from "@/lib/storage";

const Answer = dynamic(() => import("@/components/Answer"));
const ConnectionSettings = dynamic(
  () => import("@/components/ConnectionSettings"),
);
const MODES = [
  {
    id: "explain" as const,
    label: "Explain",
    icon: Lightbulb,
    title: "Find your “aha” moment",
    detail: "Break a topic into simple steps.",
    prompt: "Explain a key concept from this chapter with a simple example.",
  },
  {
    id: "revise" as const,
    label: "Revise",
    icon: BookOpen,
    title: "Keep the important bits",
    detail: "Turn a chapter into revision notes.",
    prompt: "Summarize the key concepts in this chapter for quick revision.",
  },
  {
    id: "practice" as const,
    label: "Practice",
    icon: ListChecks,
    title: "Put your learning to work",
    detail: "Try questions before the exam.",
    prompt:
      "Create 5 exam-style practice questions from this chapter with answers.",
  },
  {
    id: "flashcards" as const,
    label: "Flashcards",
    icon: Layers,
    title: "A little recall goes a long way",
    detail: "Flip through concepts at your pace.",
    prompt: "Create 5 flashcards for the important concepts in this chapter.",
  },
];
const EMPTY_CONNECTION: Connection = {
  key: "",
  model: DEFAULT_MODEL,
  connected: false,
  remember: false,
};

export default function StudyWorkspace({
  account,
}: {
  account: AccountInfo | null;
}) {
  const identity = account?.id || "guest";
  const keyStorage =
    identity === "guest" ? "ncert-key-v1" : `ncert-key-v1:${identity}`;
  const [view, setView] = useState<View>("study");
  const [documents, setDocuments] = useState<Textbook[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState("");
  const [connection, setConnection] = useState<Connection>(EMPTY_CONNECTION);
  const [serverKey, setServerKey] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Textbook | null>(null);
  const [deleteChatId, setDeleteChatId] = useState("");
  const [deleting, setDeleting] = useState("");
  const [loadingLibrary, setLoadingLibrary] = useState(true);
  const [ready, setReady] = useState(false);
  const [uploading, setUploading] = useState("");
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<StudyMode>("explain");
  const [scope, setScope] = useState("");
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [storageWarning, setStorageWarning] = useState("");
  const [retry, setRetry] = useState<{ text: string; mode: StudyMode } | null>(
    null,
  );
  const [showLatest, setShowLatest] = useState(false);
  const [hasKeyDraft, setHasKeyDraft] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const uploadLock = useRef(false);
  const sendLock = useRef(false);
  const scroll = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const active = conversations.find((c) => c.id === activeId);
  const messages = active?.messages ?? [];
  const saved = conversations.flatMap((c) =>
    c.messages
      .map((message, index) => ({ message, index, conversation: c }))
      .filter((item) => item.message.saved),
  );

  useEffect(() => {
    let alive = true;
    const controller = new AbortController();
    async function initialize() {
      // Establish the workspace cookie before any library request to avoid parallel first-visit sessions.
      let health;
      try {
        health = await readResponse(
          await fetch("/api/health", {
            signal: controller.signal,
            headers: { "X-NCERT-Identity": identity },
          }),
        );
      } catch (err) {
        if (alive)
          setError(
            err instanceof Error
              ? err.message
              : "The study service is offline.",
          );
      }
      if (!alive) return;
      const stored = loadConversations(identity);
      const initial = stored.length
        ? stored
        : [
            {
              id: crypto.randomUUID(),
              title: "New study session",
              messages: [],
            },
          ];
      setConversations(initial);
      setActiveId(initial[0].id);
      let storedModel = health?.default_model || DEFAULT_MODEL;
      try {
        storedModel = localStorage.getItem("ncert-model-v1") || storedModel;
        const key = sessionStorage.getItem(keyStorage) || "";
        setConnection({
          key,
          model: storedModel,
          connected: false,
          remember: !!key,
        });
        setHasKeyDraft(!!key);
      } catch {
        setStorageWarning(
          "Browser storage is unavailable. Sessions will last until you close this page.",
        );
      }
      setServerKey(!!health?.server_key_configured);
      try {
        const data = await readResponse(
          await fetch("/api/documents", {
            signal: controller.signal,
            headers: { "X-NCERT-Identity": identity },
          }),
        );
        if (alive) setDocuments(data.documents);
      } catch (err) {
        if (alive)
          setError(
            err instanceof Error ? err.message : "Could not open your library.",
          );
      } finally {
        if (alive) {
          setLoadingLibrary(false);
          setReady(true);
        }
      }
    }
    void initialize();
    return () => {
      alive = false;
      controller.abort();
      abort.current?.abort();
    };
  }, [identity, keyStorage]);

  useEffect(() => {
    if (!ready) return;
    const timer = setTimeout(() => {
      try {
        saveConversations(conversations, identity);
      } catch {
        setStorageWarning(
          "This browser could not save your sessions. Export a session to keep a copy.",
        );
      }
    }, 100);
    return () => clearTimeout(timer);
  }, [conversations, ready, identity]);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 5000);
    return () => clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    if (scroll.current && !showLatest)
      scroll.current.scrollTop = messages.length
        ? scroll.current.scrollHeight
        : 0;
  }, [messages.length, busy, view, showLatest]);

  useEffect(() => {
    function handleKey(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        if (!sendLock.current) {
          const id = crypto.randomUUID();
          setConversations((prev) =>
            [
              { id, title: "New study session", messages: [] },
              ...prev.filter((c) => c.messages.length),
            ].slice(0, 20),
          );
          setActiveId(id);
          setView("study");
          setInput("");
          setError("");
          setRetry(null);
          textarea.current?.focus();
        }
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  function newChat() {
    if (busy) return;
    const id = crypto.randomUUID();
    setConversations((prev) =>
      [
        { id, title: "New study session", messages: [] },
        ...prev.filter((c) => c.messages.length),
      ].slice(0, 20),
    );
    setActiveId(id);
    setInput("");
    setError("");
    setRetry(null);
    setView("study");
    setMobileOpen(false);
    setShowLatest(false);
    textarea.current?.focus();
  }
  function navigate(next: View) {
    setView(next);
    setMobileOpen(false);
    setError("");
  }
  function openConversation(id: string) {
    if (busy) return;
    setActiveId(id);
    setView("study");
    setMobileOpen(false);
    setError("");
    setRetry(null);
    setShowLatest(false);
    setInput("");
  }
  function studyTextbook(id: string) {
    setScope(id);
    setView("study");
    setMobileOpen(false);
    setNotice("Questions will use this textbook.");
    textarea.current?.focus();
  }
  function saveConnection(next: Connection) {
    setConnection(next);
    setHasKeyDraft(!!next.key);
    try {
      localStorage.setItem("ncert-model-v1", next.model);
      if (next.remember && next.key)
        sessionStorage.setItem(keyStorage, next.key);
      else sessionStorage.removeItem(keyStorage);
    } catch {
      setStorageWarning(
        "Your connection works, but this browser cannot remember it.",
      );
    }
  }
  async function refreshLibrary() {
    setLoadingLibrary(true);
    setError("");
    try {
      const data = await readResponse(
        await fetch("/api/documents", {
          headers: { "X-NCERT-Identity": identity },
        }),
      );
      setDocuments(data.documents);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not open the library.",
      );
    } finally {
      setLoadingLibrary(false);
    }
  }
  async function upload(files: File[]) {
    if (uploadLock.current || !files.length) return;
    uploadLock.current = true;
    setError("");
    let count = 0;
    const failures: string[] = [];
    try {
      for (const file of files.slice(0, 20)) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
          failures.push(`${file.name}: choose a PDF file.`);
          continue;
        }
        if (file.size > 20 * 1024 * 1024) {
          failures.push(`${file.name}: the limit is 20 MB.`);
          continue;
        }
        setUploading(file.name);
        try {
          const data = await readResponse(
            await fetch(
              `/api/upload?filename=${encodeURIComponent(file.name)}`,
              {
                method: "POST",
                headers: {
                  "Content-Type": "application/pdf",
                  "X-NCERT-Identity": identity,
                },
                body: file,
                signal: AbortSignal.timeout(115000),
              },
            ),
          );
          setDocuments((prev) => [
            data.document,
            ...prev.filter((d) => d.id !== data.document.id),
          ]);
          count++;
        } catch (err) {
          failures.push(
            `${file.name}: ${err instanceof Error ? err.message : "Upload failed."}`,
          );
        }
      }
      if (files.length > 20)
        failures.push("Only the first 20 files were processed.");
      if (count)
        setNotice(
          `${count} ${count === 1 ? "textbook is" : "textbooks are"} ready to study.`,
        );
      if (failures.length) setError(failures.join(" "));
    } finally {
      uploadLock.current = false;
      setUploading("");
    }
  }
  async function removeDocument() {
    if (!deleteTarget) return;
    const doc = deleteTarget;
    setDeleting(doc.id);
    setDeleteTarget(null);
    setError("");
    try {
      await readResponse(
        await fetch(`/api/documents/${doc.id}`, {
          method: "DELETE",
          headers: { "X-NCERT-Identity": identity },
        }),
      );
      setDocuments((prev) => prev.filter((d) => d.id !== doc.id));
      if (scope === doc.id) setScope("");
      setNotice("Textbook removed from your library.");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not remove the textbook.",
      );
    } finally {
      setDeleting("");
    }
  }
  function deleteConversation() {
    const id = deleteChatId;
    setDeleteChatId("");
    const remaining = conversations.filter((c) => c.id !== id);
    if (activeId === id) {
      const next = remaining[0] ?? {
        id: crypto.randomUUID(),
        title: "New study session",
        messages: [],
      };
      setActiveId(next.id);
      setConversations(remaining.length ? remaining : [next]);
      setRetry(null);
    } else setConversations(remaining);
    setNotice("Study session deleted.");
  }
  function toggleSaved(conversationId: string, index: number) {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === conversationId
          ? {
              ...c,
              messages: c.messages.map((m, i) =>
                i === index ? { ...m, saved: !m.saved } : m,
              ),
            }
          : c,
      ),
    );
  }
  async function send(text: string, chosenMode = mode, isRetry = false) {
    const question = text.trim();
    if (!question || sendLock.current || !ready) return;
    if (!connection.connected) {
      setSettingsOpen(true);
      return;
    }
    if (!documents.length) {
      setView("library");
      setNotice("Add a textbook to start your study session.");
      return;
    }
    if (question.length > 4000) {
      setError("Keep your question under 4,000 characters.");
      return;
    }
    sendLock.current = true;
    setBusy(true);
    setError("");
    setRetry(null);
    setInput("");
    setShowLatest(false);
    const nextMessages: Message[] = isRetry
      ? messages
      : [...messages, { role: "user", content: question }];
    const sessionId = activeId;
    setConversations((prev) =>
      prev.map((c) =>
        c.id === sessionId
          ? {
              ...c,
              title: c.messages.length ? c.title : question.slice(0, 55),
              messages: nextMessages,
            }
          : c,
      ),
    );
    const controller = new AbortController();
    abort.current = controller;
    try {
      const data = await readResponse(
        await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-NCERT-Identity": identity,
            ...(connection.key
              ? { Authorization: `Bearer ${connection.key}` }
              : {}),
          },
          body: JSON.stringify({
            messages: nextMessages.slice(-12).map(({ role, content }) => ({
              role,
              content: content.slice(0, 4000),
            })),
            model: connection.model,
            mode: chosenMode,
            document_ids: scope ? [scope] : [],
          }),
          signal: AbortSignal.any([
            controller.signal,
            AbortSignal.timeout(115000),
          ]),
        }),
      );
      const answer: Message = {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        cards: data.cards,
      };
      setConversations((prev) =>
        prev.map((c) =>
          c.id === sessionId
            ? { ...c, messages: [...c.messages, answer].slice(-40) }
            : c,
        ),
      );
    } catch (err) {
      if (controller.signal.aborted)
        setNotice("Answer stopped. You can retry your question.");
      else
        setError(
          err instanceof Error
            ? err.message
            : "Could not get an answer. Please retry.",
        );
      setRetry({ text: question, mode: chosenMode });
    } finally {
      abort.current = null;
      sendLock.current = false;
      setBusy(false);
    }
  }
  const sidebarProps = {
    view,
    onView: navigate,
    documents,
    conversations,
    activeId,
    connected: connection.connected,
    busy: busy || !ready,
    onNewChat: newChat,
    onConversation: openConversation,
    onDeleteChat: setDeleteChatId,
    onTextbook: studyTextbook,
    onSettings: () => setSettingsOpen(true),
    onHelp: () => setHelpOpen(true),
  };

  return (
    <div className="workspace">
      <a className="skip-link" href="#main-content">
        Skip to study space
      </a>
      <aside className="desktop-sidebar">
        <Sidebar {...sidebarProps} />
      </aside>
      <Dialog.Root open={mobileOpen} onOpenChange={setMobileOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="modal-overlay mobile-overlay" />
          <Dialog.Content className="mobile-sidebar">
            <Dialog.Title className="sr-only">Study navigation</Dialog.Title>
            <Dialog.Description className="sr-only">
              Your textbooks, sessions, and settings.
            </Dialog.Description>
            <Dialog.Close asChild>
              <Button
                className="mobile-close"
                variant="ghost"
                size="icon"
                aria-label="Close navigation"
              >
                <X />
              </Button>
            </Dialog.Close>
            <Sidebar {...sidebarProps} />
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <div className="main-column">
        <header className="topbar">
          <div className="topbar-title">
            <Button
              className="mobile-menu"
              variant="ghost"
              size="icon"
              aria-label="Open navigation"
              onClick={() => setMobileOpen(true)}
            >
              <Menu />
            </Button>
            <span>
              {view === "study"
                ? "Study space"
                : view === "library"
                  ? "Textbook library"
                  : "Saved answers"}
            </span>
            <span className="header-divider">/</span>
            <span className="class-label">Class 10</span>
          </div>
          <div className="header-actions">
            {view === "study" && messages.length > 0 && (
              <Button
                variant="ghost"
                size="icon"
                aria-label="Export session as Markdown"
                title="Export session"
                onClick={() => active && downloadConversation(active)}
              >
                <Download size={17} />
              </Button>
            )}
            <button
              className="connection-pill"
              onClick={() => setSettingsOpen(true)}
            >
              <span
                className={`status-dot ${connection.connected ? "connected" : ""}`}
              />
              {connection.connected ? "NVIDIA NIM" : "Connect NVIDIA"}
              <ChevronRight size={12} />
            </button>
            <ThemeToggle />
            <AccountMenu account={account} compact />
          </div>
        </header>
        <main
          id="main-content"
          className={`main-content ${view === "study" ? "chat-view" : ""}`}
        >
          {(error || storageWarning) && (
            <div className="error-banner" role="alert">
              <span>{error || storageWarning}</span>
              <button
                aria-label="Dismiss error"
                onClick={() => {
                  setError("");
                  setStorageWarning("");
                }}
              >
                <X size={15} />
              </button>
            </div>
          )}
          {notice && (
            <div className="notice" role="status">
              <Check size={15} />
              {notice}
            </div>
          )}
          {view === "library" ? (
            <Library
              documents={documents}
              uploading={uploading}
              loading={loadingLibrary}
              deleting={deleting}
              onUpload={upload}
              onStudy={studyTextbook}
              onDelete={setDeleteTarget}
              onRetry={refreshLibrary}
            />
          ) : view === "saved" ? (
            <div className="saved-page page-container">
              <div className="page-intro">
                <span className="page-icon">
                  <Bookmark size={23} />
                </span>
                <h1>Worth coming back to.</h1>
                <p>
                  Your saved explanations, revision notes, and moments of
                  clarity.
                </p>
              </div>
              {saved.length ? (
                saved.map(({ message, index, conversation }) => (
                  <div
                    className="saved-answer"
                    key={`${conversation.id}-${index}`}
                  >
                    <div className="saved-heading">
                      <button
                        onClick={() => openConversation(conversation.id)}
                        disabled={busy}
                      >
                        {conversation.title}
                        <ArrowUpRight size={14} />
                      </button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => toggleSaved(conversation.id, index)}
                      >
                        <Bookmark size={14} fill="currentColor" />
                        Unsave
                      </Button>
                    </div>
                    <Answer message={message} />
                  </div>
                ))
              ) : (
                <div className="saved-empty">
                  <Bookmark size={30} strokeWidth={1.3} />
                  <h2>A place for the useful things.</h2>
                  <p>
                    Save an answer using the bookmark below it.
                    <br />
                    Find it here whenever you need a refresher.
                  </p>
                  <Button variant="outline" onClick={() => navigate("study")}>
                    Back to study space
                  </Button>
                </div>
              )}
            </div>
          ) : (
            <>
              <div
                className="chat-scroll"
                ref={scroll}
                onScroll={() => {
                  const el = scroll.current;
                  if (el)
                    setShowLatest(
                      el.scrollHeight - el.scrollTop - el.clientHeight > 160,
                    );
                }}
              >
                {messages.length === 0 ? (
                  <div className="welcome">
                    <div className="welcome-mark">
                      <LogoMark />
                    </div>
                    <p className="welcome-greeting">
                      A fresh page, a clearer mind.
                    </p>
                    <h1>
                      Make sense of your
                      <br />
                      next chapter.
                    </h1>
                    <p className="welcome-description">
                      Big questions. Small steps. A study companion
                      <br className="desktop-break" /> grounded in your NCERT
                      textbooks.
                    </p>
                    <div className="starter-grid">
                      {MODES.map(
                        ({ id, icon: Icon, title, detail, prompt }) => (
                          <button
                            className={`starter starter-${id}`}
                            key={id}
                            disabled={busy || !ready}
                            onClick={() => {
                              setMode(id);
                              setInput(prompt);
                              textarea.current?.focus();
                            }}
                          >
                            <span className="starter-icon">
                              <Icon size={18} strokeWidth={1.7} />
                            </span>
                            <span>
                              <strong>{title}</strong>
                              <small>{detail}</small>
                            </span>
                            <ArrowUpRight size={15} />
                          </button>
                        ),
                      )}
                    </div>
                    {(!connection.connected || !documents.length) && (
                      <div className="setup-strip">
                        <span>Make yourself at home</span>
                        <button
                          className={connection.connected ? "complete" : ""}
                          onClick={() => setSettingsOpen(true)}
                        >
                          <span className="step-number">
                            {connection.connected ? <Check size={12} /> : "1"}
                          </span>
                          {connection.connected
                            ? "AI connected"
                            : "Connect your key"}
                          <ChevronRight size={12} />
                        </button>
                        <button
                          className={documents.length ? "complete" : ""}
                          onClick={() => navigate("library")}
                        >
                          <span className="step-number">
                            {documents.length ? <Check size={12} /> : "2"}
                          </span>
                          {documents.length
                            ? "Textbook ready"
                            : "Add a textbook"}
                          <ChevronRight size={12} />
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="conversation" aria-label="Study conversation">
                    {messages.map((message, index) =>
                      message.role === "user" ? (
                        <div
                          className="user-message"
                          key={`${activeId}-${index}`}
                        >
                          <span className="sr-only">You: </span>
                          {message.content}
                        </div>
                      ) : (
                        <div
                          className="answer-wrap"
                          key={`${activeId}-${index}`}
                        >
                          <Answer message={message} />
                          <Button
                            className={`save-answer ${message.saved ? "is-saved" : ""}`}
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleSaved(activeId, index)}
                            aria-label={
                              message.saved ? "Unsave answer" : "Save answer"
                            }
                            aria-pressed={!!message.saved}
                          >
                            <Bookmark
                              size={14}
                              fill={message.saved ? "currentColor" : "none"}
                            />
                            {message.saved ? "Saved" : "Save"}
                          </Button>
                        </div>
                      ),
                    )}
                    {busy && (
                      <div className="thinking" role="status">
                        <LoaderCircle size={17} className="spin" />
                        <span>Reading your textbook and thinking…</span>
                      </div>
                    )}
                    {retry && (
                      <div className="retry-row">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => send(retry.text, retry.mode, true)}
                        >
                          <RotateCcw size={14} />
                          Retry question
                        </Button>
                        <span>Your question is still here.</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
              {showLatest && (
                <Button
                  className="jump-latest"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    if (scroll.current)
                      scroll.current.scrollTop = scroll.current.scrollHeight;
                    setShowLatest(false);
                  }}
                >
                  <ArrowDown size={14} />
                  Latest
                </Button>
              )}
              <div className="composer-area">
                <form
                  className="composer"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void send(input);
                  }}
                >
                  <div className="composer-scope">
                    <BookOpen size={13} />
                    <select
                      aria-label="Textbooks used for answers"
                      value={scope}
                      disabled={busy}
                      onChange={(e) => setScope(e.target.value)}
                    >
                      <option value="">
                        {documents.length
                          ? `All textbooks (${documents.length})`
                          : "Your textbooks"}
                      </option>
                      {documents.map((doc) => (
                        <option key={doc.id} value={doc.id}>
                          {doc.name}
                        </option>
                      ))}
                    </select>
                    {uploading && (
                      <span className="upload-indicator">
                        <LoaderCircle size={12} className="spin" />
                        Preparing PDF…
                      </span>
                    )}
                  </div>
                  <textarea
                    ref={textarea}
                    aria-label="Ask your textbook a question"
                    placeholder={
                      mode === "flashcards"
                        ? "What topic would you like to remember?"
                        : "What would you like to understand?"
                    }
                    value={input}
                    maxLength={4000}
                    rows={2}
                    onChange={(e) => {
                      setInput(e.target.value);
                      e.target.style.height = "auto";
                      e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
                    }}
                    onKeyDown={(e) => {
                      if (
                        e.key === "Enter" &&
                        !e.shiftKey &&
                        !e.nativeEvent.isComposing
                      ) {
                        e.preventDefault();
                        void send(input);
                      }
                    }}
                  />
                  <div className="composer-bottom">
                    <div
                      className="mode-switch"
                      role="group"
                      aria-label="Study mode"
                    >
                      {MODES.map(({ id, label, icon: Icon }) => (
                        <button
                          type="button"
                          key={id}
                          aria-pressed={mode === id}
                          disabled={busy}
                          onClick={() => setMode(id)}
                        >
                          <Icon size={14} />
                          <span>{label}</span>
                        </button>
                      ))}
                    </div>
                    <div className="send-actions">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        aria-label="Attach a textbook PDF"
                        title="Attach a PDF"
                        disabled={!!uploading || loadingLibrary}
                        onClick={() => fileInput.current?.click()}
                      >
                        <Paperclip size={17} />
                      </Button>
                      {busy ? (
                        <Button
                          type="button"
                          size="icon"
                          aria-label="Stop answer"
                          className="send-button"
                          onClick={() => abort.current?.abort()}
                        >
                          <Square size={15} fill="currentColor" />
                        </Button>
                      ) : (
                        <Button
                          type="submit"
                          size="icon"
                          className="send-button"
                          aria-label="Send question"
                          disabled={!input.trim() || !ready}
                        >
                          <ArrowUp size={19} />
                        </Button>
                      )}
                    </div>
                  </div>
                </form>
                <p className="composer-note">
                  <span>
                    Learn with curiosity. Check answers against your textbook.
                  </span>
                  <span className="enter-hint">Enter to send</span>
                </p>
              </div>
            </>
          )}
        </main>
      </div>
      <input
        ref={fileInput}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        hidden
        aria-label="Attach PDFs"
        onChange={(e) => {
          void upload(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />
      {settingsOpen && (
        <ConnectionSettings
          open={settingsOpen}
          onOpenChange={setSettingsOpen}
          connection={connection}
          serverKey={serverKey}
          identity={identity}
          onSave={saveConnection}
        />
      )}
      <Modal
        open={helpOpen}
        onOpenChange={setHelpOpen}
        title="A little guide to getting started"
        description="Your own textbooks, a bit of curiosity, and you."
      >
        <ol className="quick-guide">
          <li>
            <KeyHint />
            <div>
              <strong>Connect your NVIDIA key</strong>
              <p>
                Open Settings, paste your API key, choose a model, and test the
                connection.{" "}
                {hasKeyDraft
                  ? "Your saved key is ready to test again."
                  : "Get a developer key from the NVIDIA API Catalog."}
              </p>
            </div>
          </li>
          <li>
            <BookOpen size={21} />
            <div>
              <strong>Add your study material</strong>
              <p>
                Upload a text-based NCERT PDF. Choose a chapter-sized file for
                the most focused results. You can select a specific book above
                the question box.
              </p>
            </div>
          </li>
          <li>
            <Lightbulb size={21} />
            <div>
              <strong>Ask, revise, practice, remember</strong>
              <p>
                Choose a study mode. Ask a specific topic question, or use a
                starter. Expand textbook passages to see the source and PDF
                page.
              </p>
            </div>
          </li>
          <li>
            <Bookmark size={21} />
            <div>
              <strong>Keep what helps</strong>
              <p>
                Save useful answers, revisit your sessions, or export a chat as
                Markdown using the download button. Up to 20 recent sessions are
                kept in this browser.
              </p>
            </div>
          </li>
        </ol>
        <p className="guide-tip">
          Press ⌘/Ctrl K for a new session. Shift + Enter adds a new line.
        </p>
        <Button onClick={() => setHelpOpen(false)}>Got it, let’s learn</Button>
      </Modal>
      <Modal
        open={!!deleteTarget}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
        title="Remove this textbook?"
        description={`“${deleteTarget?.name ?? ""}” will be removed from your library and search index. You can upload it again later.`}
      >
        <div className="modal-actions">
          <Button variant="outline" onClick={() => setDeleteTarget(null)}>
            Keep textbook
          </Button>
          <Button variant="destructive" onClick={removeDocument}>
            Remove textbook
          </Button>
        </div>
      </Modal>
      <Modal
        open={!!deleteChatId}
        onOpenChange={(open) => {
          if (!open) setDeleteChatId("");
        }}
        title="Delete this study session?"
        description="The conversation and its saved answers will be removed from this browser. Export it first if you want to keep a copy."
      >
        <div className="modal-actions">
          <Button variant="outline" onClick={() => setDeleteChatId("")}>
            Keep session
          </Button>
          <Button variant="destructive" onClick={deleteConversation}>
            Delete session
          </Button>
        </div>
      </Modal>
    </div>
  );
}

function KeyHint() {
  return <span className="key-hint">N</span>;
}
