"use client";

import { useState } from "react";
import { ArrowLeft, ArrowRight, Check, RotateCcw } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { Flashcard } from "@/lib/types";
import { Button } from "./ui/button";

export default function Flashcards({ cards }: { cards: Flashcard[] }) {
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [known, setKnown] = useState<number[]>([]);
  const card = cards[index];
  if (!card) return null;
  function move(next: number) {
    setIndex(next);
    setFlipped(false);
  }
  return (
    <section className="flashcard-review" aria-label="Flashcard review">
      <div className="flashcard-top">
        <span>Recall & remember</span>
        <span>
          {index + 1} / {cards.length}
        </span>
      </div>
      <div className={`flashcard ${flipped ? "flipped" : ""}`}>
        <span className="flashcard-side">
          {flipped ? "Answer" : "Question"}
        </span>
        <div className="markdown" aria-live="polite">
          <ReactMarkdown>{flipped ? card.answer : card.question}</ReactMarkdown>
        </div>
        <Button variant="outline" onClick={() => setFlipped(!flipped)}>
          {flipped ? "Show question" : "Reveal answer"}
        </Button>
      </div>
      <div className="flashcard-controls">
        <Button
          variant="ghost"
          size="icon"
          aria-label="Previous flashcard"
          disabled={index === 0}
          onClick={() => move(index - 1)}
        >
          <ArrowLeft />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          aria-pressed={known.includes(index)}
          onClick={() =>
            setKnown((prev) =>
              prev.includes(index)
                ? prev.filter((n) => n !== index)
                : [...prev, index],
            )
          }
        >
          <Check size={14} />
          {known.includes(index) ? "Remembered" : "I know this"}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Next flashcard"
          disabled={index === cards.length - 1}
          onClick={() => move(index + 1)}
        >
          <ArrowRight />
        </Button>
      </div>
      <div className="flashcard-progress">
        <progress
          max={cards.length}
          value={known.length}
          aria-label={`${known.length} of ${cards.length} cards remembered`}
        />
        <span>{known.length} remembered</span>
        <button
          aria-label="Restart flashcard review"
          onClick={() => {
            setKnown([]);
            move(0);
          }}
        >
          <RotateCcw size={13} />
        </button>
      </div>
    </section>
  );
}
