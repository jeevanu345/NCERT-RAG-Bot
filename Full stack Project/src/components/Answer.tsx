"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, Check, ChevronDown, Copy } from "lucide-react";
import type { Message } from "@/lib/types";
import { Button } from "./ui/button";
import Flashcards from "./Flashcards";

export default function Answer({ message }: { message: Message }) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const sources = message.sources ?? [];
  async function copy() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setCopyError(false);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopyError(true);
    }
  }
  return (
    <article className="assistant-message">
      <div className="answer-label">
        <span className="mini-brand">
          <BookOpen size={14} />
        </span>
        NCERT Tutor
      </div>
      {message.cards?.length ? (
        <Flashcards cards={message.cards} />
      ) : (
        <div className="markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>
      )}
      {sources.length > 0 && (
        <details className="source-list">
          <summary>
            <BookOpen size={14} />
            {sources.length} textbook{" "}
            {sources.length === 1 ? "passage" : "passages"}
            <ChevronDown size={14} />
          </summary>
          <div className="source-items">
            {sources.map((source) => (
              <details className="source-item" key={source.citation}>
                <summary>
                  <span className="source-number">{source.citation}</span>
                  <span>
                    {source.name}
                    <small>
                      PDF page {source.page}
                      {source.cited
                        ? " · Cited in answer"
                        : " · Retrieved context"}
                    </small>
                  </span>
                  <ChevronDown size={14} />
                </summary>
                <p>{source.excerpt}</p>
              </details>
            ))}
          </div>
        </details>
      )}
      <div className="answer-actions">
        <Button
          size="sm"
          variant="ghost"
          onClick={copy}
          aria-label="Copy answer"
        >
          {copied ? <Check /> : <Copy />}
          {copied ? "Copied" : "Copy"}
        </Button>
        {copyError && (
          <span role="status">Select the answer to copy it manually.</span>
        )}
      </div>
    </article>
  );
}
