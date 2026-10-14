"use client";

import { useRef, useState } from "react";
import {
  ArrowUpRight,
  BookOpen,
  Check,
  FileText,
  LoaderCircle,
  Search,
  Trash2,
  Upload,
} from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import type { Textbook } from "@/lib/types";

export default function Library({
  documents,
  uploading,
  loading,
  deleting,
  onUpload,
  onStudy,
  onDelete,
  onRetry,
}: {
  documents: Textbook[];
  uploading: string;
  loading: boolean;
  deleting: string;
  onUpload: (files: File[]) => void;
  onStudy: (id: string) => void;
  onDelete: (doc: Textbook) => void;
  onRetry: () => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [search, setSearch] = useState("");
  const [dragging, setDragging] = useState(false);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [sampleError, setSampleError] = useState("");
  async function useSample() {
    setSampleLoading(true);
    setSampleError("");
    try {
      const response = await fetch("/sample-study-notes.pdf");
      if (!response.ok)
        throw new Error("Could not open sample notes. Try uploading a PDF.");
      onUpload([
        new File([await response.blob()], "Sample science notes.pdf", {
          type: "application/pdf",
        }),
      ]);
    } catch (error) {
      setSampleError(
        error instanceof Error ? error.message : "Could not open sample notes.",
      );
    } finally {
      setSampleLoading(false);
    }
  }
  const filtered = documents.filter((doc) =>
    doc.name.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <div className="library-page page-container">
      <div className="page-intro">
        <span className="page-icon">
          <BookOpen size={23} />
        </span>
        <h1>A shelf for your curiosity.</h1>
        <p>
          Add a textbook or chapter. Every explanation starts with your own
          study material.
        </p>
      </div>
      <div
        className={`upload-zone ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!uploading) onUpload(Array.from(e.dataTransfer.files));
        }}
        aria-busy={!!uploading}
      >
        {uploading ? (
          <LoaderCircle size={25} className="spin" />
        ) : (
          <Upload size={25} strokeWidth={1.5} />
        )}
        <h2>
          {uploading ? "Preparing your textbook…" : "Drop your PDFs here"}
        </h2>
        <p>
          {uploading || "Text-based PDFs, up to 20 MB each. Up to 20 books."}
        </p>
        <Button
          variant="outline"
          disabled={!!uploading || loading}
          onClick={() => input.current?.click()}
        >
          {uploading ? "Extracting & indexing…" : "Choose PDF files"}
        </Button>
        <input
          ref={input}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          aria-label="Upload textbook PDFs"
          onChange={(e) => {
            onUpload(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
      </div>
      {!documents.length && (
        <p className="sample-note">
          No textbook handy?{" "}
          <button
            className="text-link"
            disabled={!!uploading || loading || sampleLoading}
            onClick={useSample}
          >
            {sampleLoading ? "Opening sample…" : "Try sample science notes"}
          </button>
        </p>
      )}
      {sampleError && (
        <p className="inline-error" role="alert">
          {sampleError}
        </p>
      )}
      <div className="library-toolbar">
        <h2>
          Your library <span>{documents.length}</span>
        </h2>
        <div className="search-field">
          <Search size={15} />
          <Input
            aria-label="Search textbooks"
            placeholder="Find a textbook…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>
      {loading ? (
        <div className="library-empty">
          <LoaderCircle className="spin" />
          <p>Opening your library…</p>
        </div>
      ) : filtered.length ? (
        <div className="document-list">
          {filtered.map((doc) => (
            <div className="document-row" key={doc.id}>
              <span className="document-icon">
                <FileText size={22} strokeWidth={1.5} />
              </span>
              <div className="document-info">
                <h3>{doc.name}</h3>
                <p>
                  {doc.pages} {doc.pages === 1 ? "page" : "pages"}
                  <span className="ready-label">
                    <Check size={12} />
                    Ready to study
                  </span>
                </p>
              </div>
              <Button variant="ghost" size="sm" onClick={() => onStudy(doc.id)}>
                Study this
                <ArrowUpRight size={14} />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="danger-hover"
                aria-label={`Remove ${doc.name}`}
                disabled={!!deleting || !!uploading}
                onClick={() => onDelete(doc)}
              >
                {deleting === doc.id ? (
                  <LoaderCircle className="spin" />
                ) : (
                  <Trash2 size={16} />
                )}
              </Button>
            </div>
          ))}
        </div>
      ) : (
        <div className="library-empty">
          <BookOpen size={24} />
          <p>
            {search
              ? "No textbooks match your search."
              : "Your shelf is waiting for its first book."}
          </p>
          {!search && (
            <button className="text-link" onClick={onRetry}>
              Refresh library
            </button>
          )}
        </div>
      )}
      <p className="library-note">
        Your library is saved on this computer for this browser. Scanned PDFs
        need text recognition before uploading.
      </p>
    </div>
  );
}
