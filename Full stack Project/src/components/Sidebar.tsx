"use client";

import { useState } from "react";
import Link from "next/link";
import Logo from "./brand/Logo";
import {
  BookMarked,
  BookOpen,
  Bookmark,
  ChevronRight,
  CircleHelp,
  MessageCircle,
  Plus,
  Search,
  Settings2,
  Trash2,
} from "lucide-react";
import { Button } from "./ui/button";
import type { Conversation, Textbook } from "@/lib/types";

export type View = "study" | "library" | "saved";
export type SidebarProps = {
  view: View;
  onView: (view: View) => void;
  documents: Textbook[];
  conversations: Conversation[];
  activeId: string;
  connected: boolean;
  busy: boolean;
  onNewChat: () => void;
  onConversation: (id: string) => void;
  onDeleteChat: (id: string) => void;
  onTextbook: (id: string) => void;
  onSettings: () => void;
  onHelp: () => void;
};

export default function Sidebar(props: SidebarProps) {
  const [search, setSearch] = useState("");
  const chats = props.conversations.filter(
    (c) =>
      c.messages.length > 0 &&
      c.title.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <div className="sidebar-content">
      <Link href="/" className="brand" aria-label="NCERT Tutor home">
        <Logo />
      </Link>
      <Button
        className="new-chat"
        onClick={props.onNewChat}
        disabled={props.busy}
      >
        <Plus size={17} />
        New study session<span className="shortcut">⌘ K</span>
      </Button>
      <nav className="main-nav" aria-label="Workspace">
        <button
          aria-current={props.view === "study" ? "page" : undefined}
          onClick={() => props.onView("study")}
        >
          <MessageCircle size={17} />
          Study space
        </button>
        <button
          aria-current={props.view === "library" ? "page" : undefined}
          onClick={() => props.onView("library")}
        >
          <BookMarked size={17} />
          Textbook library
          <span className="nav-count">{props.documents.length}</span>
        </button>
        <button
          aria-current={props.view === "saved" ? "page" : undefined}
          onClick={() => props.onView("saved")}
        >
          <Bookmark size={17} />
          Saved answers
        </button>
      </nav>
      <div className="sidebar-section">
        <div className="section-heading">
          <h2>Your textbooks</h2>
          <button
            aria-label="Manage textbooks"
            onClick={() => props.onView("library")}
          >
            <Plus size={15} />
          </button>
        </div>
        {props.documents.length ? (
          props.documents.slice(0, 4).map((doc) => (
            <button
              className="textbook-shortcut"
              key={doc.id}
              onClick={() => props.onTextbook(doc.id)}
            >
              <span className="small-book">
                <BookOpen size={15} />
              </span>
              <span>
                {doc.name.replace(/\.pdf$/i, "")}
                <small>
                  {doc.pages} {doc.pages === 1 ? "page" : "pages"}
                </small>
              </span>
            </button>
          ))
        ) : (
          <button
            className="sidebar-empty"
            onClick={() => props.onView("library")}
          >
            <BookOpen size={19} />
            <span>
              Your next chapter starts here.
              <small>
                Add your first textbook <ChevronRight size={12} />
              </small>
            </span>
          </button>
        )}
      </div>
      <div className="sidebar-section recent-section">
        <div className="section-heading">
          <h2>Recent sessions</h2>
          <span>
            {props.conversations.filter((c) => c.messages.length).length || ""}
          </span>
        </div>
        {props.conversations.some((c) => c.messages.length) && (
          <div className="history-search">
            <Search size={13} />
            <input
              aria-label="Search study sessions"
              placeholder="Find a session…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        )}
        <div className="history-list">
          {chats.length ? (
            chats.map((chat) => (
              <div
                key={chat.id}
                className={`history-row ${chat.id === props.activeId && props.view === "study" ? "active" : ""}`}
              >
                <button
                  disabled={props.busy}
                  onClick={() => props.onConversation(chat.id)}
                >
                  <MessageCircle size={14} />
                  <span>{chat.title}</span>
                </button>
                <button
                  className="delete-chat"
                  aria-label={`Delete session: ${chat.title}`}
                  disabled={props.busy}
                  onClick={() => props.onDeleteChat(chat.id)}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))
          ) : (
            <p className="empty-history">
              {search ? "No matching sessions." : "Good questions belong here."}
            </p>
          )}
        </div>
      </div>
      <div className="sidebar-bottom">
        <button className="connection-card" onClick={props.onSettings}>
          <span
            className={`status-dot ${props.connected ? "connected" : ""}`}
          />
          <span>
            <strong>
              {props.connected ? "NVIDIA connected" : "Connect your AI"}
            </strong>
            <small>
              {props.connected
                ? "Your key. Your study space."
                : "Bring your NVIDIA API key"}
            </small>
          </span>
          <ChevronRight size={15} />
        </button>
        <div className="sidebar-utilities">
          <button onClick={props.onSettings}>
            <Settings2 size={15} />
            Settings
          </button>
          <button onClick={props.onHelp}>
            <CircleHelp size={15} />
            Quick guide
          </button>
        </div>
        <p className="sidebar-footnote">Made for your next “I get it.”</p>
      </div>
    </div>
  );
}
