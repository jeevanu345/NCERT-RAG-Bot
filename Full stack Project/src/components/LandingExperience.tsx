"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Bookmark,
  Check,
  CheckCheck,
  ChevronDown,
  FileText,
  KeyRound,
  Layers,
  Lightbulb,
  ListChecks,
  Menu,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import Logo, { LogoMark } from "./brand/Logo";
import ThemeToggle from "./ThemeToggle";
import AccountMenu from "./AccountMenu";
import type { AccountInfo } from "@/lib/auth-config";

type DemoMode = "explain" | "revise" | "practice" | "flashcards";
const modes = [
  {
    id: "explain" as const,
    title: "Understand it",
    label: "Explain",
    icon: Lightbulb,
    detail:
      "A patient explanation, a familiar example, and a place to ask “why?” again.",
  },
  {
    id: "revise" as const,
    title: "Find the essentials",
    label: "Revise",
    icon: BookOpen,
    detail:
      "Turn the ideas in your chapter into clear notes, definitions, and a focused recap.",
  },
  {
    id: "practice" as const,
    title: "Give it a go",
    label: "Practice",
    icon: ListChecks,
    detail:
      "Work through questions before looking at the answers. A little practice goes a long way.",
  },
  {
    id: "flashcards" as const,
    title: "Make it stick",
    label: "Flashcards",
    icon: Layers,
    detail:
      "Reveal, recall, repeat. Keep track of the concepts you remember as you work through a set.",
  },
];

function ProductDemo({
  large = false,
  initialMode = "explain",
  onModeChange,
}: {
  large?: boolean;
  initialMode?: DemoMode;
  onModeChange?: (mode: DemoMode) => void;
}) {
  const [mode, setMode] = useState<DemoMode>(initialMode);
  const [flipped, setFlipped] = useState(false);
  const [choice, setChoice] = useState("");
  const [remembered, setRemembered] = useState(false);
  return (
    <div className={`product-demo ${large ? "large-demo" : ""}`}>
      <div className="demo-chrome">
        <span className="demo-chrome-mark">
          <LogoMark />
        </span>
        <span>Your study space</span>
        <span className="demo-sample-label">Interactive example</span>
      </div>
      <div className="demo-document">
        <span>
          <FileText size={17} />
        </span>
        <div>
          <strong>Electricity</strong>
          <small>Sample science notes · PDF page 1</small>
        </div>
        <span className="demo-ready">
          <Check size={11} />
          Ready
        </span>
      </div>
      <div
        className="demo-modes"
        role="group"
        aria-label={large ? "Explore study modes" : "Preview study modes"}
      >
        {modes.map((m) => (
          <button
            type="button"
            key={m.id}
            aria-pressed={mode === m.id}
            onClick={() => {
              setMode(m.id);
              onModeChange?.(m.id);
              setFlipped(false);
              setChoice("");
            }}
          >
            <m.icon size={13} />
            {m.label}
          </button>
        ))}
      </div>
      <div className="demo-content" key={mode}>
        {mode === "explain" && (
          <>
            <div className="demo-question">
              Can you explain Ohm’s law simply?
            </div>
            <div className="demo-answer-label">
              <LogoMark />
              NCERT Tutor
            </div>
            <h3>A useful relationship.</h3>
            <p>
              At a constant temperature, the potential difference across a
              conductor equals its current multiplied by its resistance.{" "}
              <span className="demo-citation">1</span>
            </p>
            <div className="demo-formula">
              <span>V = I × R</span>
              <small>voltage = current × resistance</small>
            </div>
            <p className="demo-example">
              For example, 6 volts across 3 ohms produces a current of{" "}
              <strong>2 amperes.</strong>
            </p>
          </>
        )}
        {mode === "revise" && (
          <>
            <div className="demo-question">
              Give me a quick recap of this page.
            </div>
            <div className="demo-answer-label">
              <LogoMark />
              Your revision notes
            </div>
            <h3>The ideas to take with you.</h3>
            <ul className="demo-revision">
              <li>
                <Check size={14} />
                <span>
                  <strong>Current</strong> is the rate of flow of charge. I = Q
                  / t.
                </span>
              </li>
              <li>
                <Check size={14} />
                <span>
                  <strong>Potential difference</strong> is work done per unit
                  charge. V = W / Q.
                </span>
              </li>
              <li>
                <Check size={14} />
                <span>
                  <strong>Ohm’s law</strong> connects voltage, current, and
                  resistance. V = I × R.
                </span>
              </li>
            </ul>
            <span className="demo-note">
              From the sample’s first page. Keep the textbook close.
            </span>
          </>
        )}
        {mode === "practice" && (
          <>
            <div className="demo-question">
              Let me try an electricity question.
            </div>
            <div className="demo-answer-label">
              <ListChecks size={16} />A little practice
            </div>
            <h3>Put your understanding to work.</h3>
            <p>
              A potential difference of 6 V is applied across a 3 Ω resistor.
              What current flows?
            </p>
            <div className="demo-choices">
              {["1 A", "2 A", "3 A"].map((value) => (
                <button
                  type="button"
                  key={value}
                  aria-pressed={choice === value}
                  onClick={() => setChoice(value)}
                >
                  {value}
                  {choice === value && <Check size={13} />}
                </button>
              ))}
            </div>
            <p className="demo-feedback" aria-live="polite">
              {choice
                ? choice === "2 A"
                  ? "That’s it. I = V / R = 6 / 3 = 2 A."
                  : "Try once more. Divide the voltage by the resistance."
                : "Choose an answer to check your understanding."}
            </p>
          </>
        )}
        {mode === "flashcards" && (
          <>
            <div className="demo-answer-label">
              <Layers size={16} />A moment for recall<span>1 / 5</span>
            </div>
            <div className="demo-flashcard">
              <small>{flipped ? "Answer" : "Question"}</small>
              <h3>
                {flipped
                  ? "V = I × R"
                  : "What relationship does Ohm’s law describe?"}
              </h3>
              <p>
                {flipped
                  ? "At constant temperature, voltage equals current multiplied by resistance. [1]"
                  : "Take your time. Bring the idea to mind."}
              </p>
              <button
                type="button"
                className="demo-reveal"
                onClick={() => setFlipped(!flipped)}
              >
                {flipped ? "Show question" : "Reveal answer"}
              </button>
            </div>
            <button
              type="button"
              className="demo-remember"
              aria-pressed={remembered}
              onClick={() => setRemembered(!remembered)}
            >
              <CheckCheck size={14} />
              {remembered ? "One more idea remembered." : "I know this one"}
            </button>
          </>
        )}
      </div>
      <details className="demo-source">
        <summary>
          <BookOpen size={12} />
          See the textbook passage<span>PDF page 1</span>
          <ChevronDown size={12} />
        </summary>
        <p>
          “Ohm’s law: at constant temperature, voltage equals current times
          resistance. V = I R. Resistance is measured in ohms.”
          <small>
            Original sample science notes included with this project.
          </small>
        </p>
      </details>
      <div className="demo-footer">
        <ShieldCheck size={11} />
        An example from our sample notes. No API request.
      </div>
    </div>
  );
}

export default function LandingExperience({
  account,
  guestAllowed,
}: {
  account: AccountInfo | null;
  guestAllowed: boolean;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeMode, setActiveMode] = useState<DemoMode>("explain");
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (
      window.matchMedia("(prefers-reduced-motion: reduce)").matches ||
      !("IntersectionObserver" in window)
    )
      return;
    const observer = new IntersectionObserver(
      (entries) =>
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.remove("reveal-pending");
            observer.unobserve(entry.target);
          }
        }),
      { threshold: 0.08, rootMargin: "0px 0px -25px 0px" },
    );
    root.current?.querySelectorAll("[data-reveal]").forEach((element) => {
      if (element.getBoundingClientRect().top > innerHeight)
        element.classList.add("reveal-pending");
      observer.observe(element);
    });
    return () => observer.disconnect();
  }, []);
  return (
    <div className="landing" ref={root}>
      <a href="#landing-main" className="skip-link">
        Skip to content
      </a>
      <header className="site-header">
        <div className="site-header-inner">
          <Link href="/" aria-label="NCERT Tutor home">
            <Logo />
          </Link>
          <nav
            className={menuOpen ? "site-nav is-open" : "site-nav"}
            id="site-navigation"
            aria-label="Main navigation"
            onKeyDown={(event) => {
              if (event.key === "Escape" && menuOpen) {
                setMenuOpen(false);
                root.current
                  ?.querySelector<HTMLButtonElement>(".site-menu-toggle")
                  ?.focus();
              }
            }}
          >
            <a href="#ways-to-learn" onClick={() => setMenuOpen(false)}>
              Ways to learn
            </a>
            <a href="#how-it-works" onClick={() => setMenuOpen(false)}>
              How it works
            </a>
            <a href="#your-space" onClick={() => setMenuOpen(false)}>
              Your space
            </a>
            <a href="#questions" onClick={() => setMenuOpen(false)}>
              Questions
            </a>
          </nav>
          <div className="site-actions">
            <ThemeToggle />
            <AccountMenu account={account} />
            <Link href="/app" className="site-start">
              Start studying
              <ArrowUpRight size={14} />
            </Link>
            <button
              className="site-menu-toggle"
              aria-label={menuOpen ? "Close menu" : "Open menu"}
              aria-expanded={menuOpen}
              aria-controls="site-navigation"
              onKeyDown={(event) => {
                if (event.key === "Escape") setMenuOpen(false);
              }}
              onClick={() => setMenuOpen(!menuOpen)}
            >
              {menuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>
      </header>
      <main id="landing-main">
        <section className="landing-hero site-width">
          <div className="hero-copy">
            <span className="hero-note">
              <span />
              Made for your next “I get it.”
            </span>
            <h1>A clearer way to meet your next chapter.</h1>
            <p>
              Your NCERT textbooks. Your own questions. A thoughtful study
              companion that helps the ideas fall into place, one small step at
              a time.
            </p>
            <div className="hero-actions">
              <Link href="/app" className="hero-primary">
                Find your starting point
                <ArrowUpRight size={18} />
              </Link>
              <a href="#ways-to-learn" className="hero-secondary">
                Take a little look
                <ArrowDown size={15} />
              </a>
            </div>
            <div className="hero-details">
              <span>
                <Check size={13} />
                Your NVIDIA key
              </span>
              <span>
                <Check size={13} />
                Your own textbooks
              </span>
              <span>
                <Check size={13} />
                Your pace
              </span>
            </div>
            <a className="hero-gentle-link" href="#how-it-works">
              <span className="mini-paper">
                <BookOpen size={14} />
              </span>
              A few minutes to set up. A little more room to learn.
              <ArrowRight size={13} />
            </a>
          </div>
          <div className="hero-product">
            <div className="hero-product-caption">
              <span className="handwritten-arrow">↘</span>From a question to a
              little clarity.
            </div>
            <ProductDemo />
            <div className="hero-source-chip">
              <span>
                <Bookmark size={14} />
              </span>
              <div>
                <strong>Always a place to look back.</strong>
                <small>Page references keep the textbook in view.</small>
              </div>
            </div>
          </div>
        </section>
        <div className="landing-principles site-width">
          <span>Built around the way you study</span>
          <div>
            <BookOpen size={18} />
            <strong>Grounded in your books</strong>
          </div>
          <div>
            <Layers size={18} />
            <strong>Four ways to learn</strong>
          </div>
          <div>
            <ShieldCheck size={18} />
            <strong>A key you control</strong>
          </div>
        </div>
        <section
          className="learning-section site-width section-space"
          id="ways-to-learn"
          data-reveal
        >
          <div className="section-intro">
            <div>
              <span className="section-kicker">
                Understanding has more than one doorway.
              </span>
              <h2>
                One chapter.
                <br />
                Your way into it.
              </h2>
            </div>
            <p>
              Some days you need the big picture. Other days, a quick recap or a
              little practice. Make room for all of them.
            </p>
          </div>
          <div className="learning-grid">
            <div className="learning-options">
              {modes.map(({ id, title, label, icon: Icon, detail }) => (
                <button
                  type="button"
                  className={
                    activeMode === id
                      ? "learning-option active"
                      : "learning-option"
                  }
                  aria-pressed={activeMode === id}
                  key={id}
                  onClick={() => setActiveMode(id)}
                >
                  <span className={`learning-icon mode-${id}`}>
                    <Icon size={21} />
                  </span>
                  <span>
                    <small>{label}</small>
                    <strong>{title}</strong>
                    {activeMode === id && <p>{detail}</p>}
                  </span>
                  <ArrowUpRight size={16} />
                </button>
              ))}
              <p className="learning-footnote">
                Try the buttons in the example. A little curiosity is all you
                need.
              </p>
            </div>
            <div className="learning-demo" key={activeMode}>
              <ModePreview mode={activeMode} onModeChange={setActiveMode} />
            </div>
          </div>
        </section>
        <section className="steps-section section-space" id="how-it-works">
          <div className="site-width" data-reveal>
            <div className="section-intro centered">
              <span className="section-kicker">A simple beginning.</span>
              <h2>Your next chapter starts here.</h2>
              <p>
                A textbook, your NVIDIA key, and a question that’s been on your
                mind.
              </p>
            </div>
            <div className="steps-grid">
              <article>
                <div className="step-visual key-visual">
                  <KeyRound size={28} />
                  <span className="key-dots">•••• •••• ••••</span>
                  <span className="step-check">
                    <Check size={12} />
                  </span>
                </div>
                <span className="step-counter">01</span>
                <h3>Bring your key.</h3>
                <p>
                  Connect your NVIDIA NIM API key in Settings. Test the
                  connection and choose a model that works for your account.
                </p>
                <a
                  href="https://build.nvidia.com/meta/llama-3_3-70b-instruct"
                  target="_blank"
                  rel="noreferrer"
                >
                  Get a developer key
                  <ArrowUpRight size={13} />
                </a>
              </article>
              <article>
                <div className="step-visual upload-visual">
                  <div className="paper-sheet">
                    <FileText size={24} />
                    <span />
                    <span />
                    <span />
                  </div>
                  <span className="upload-badge">
                    <Upload size={15} />
                  </span>
                </div>
                <span className="step-counter">02</span>
                <h3>Open your textbook.</h3>
                <p>
                  Add a text-based PDF or a focused chapter. The library makes
                  it easy to find the book you want to study.
                </p>
                <Link href="/app">
                  Or try the sample notes
                  <ArrowUpRight size={13} />
                </Link>
              </article>
              <article>
                <div className="step-visual question-visual">
                  <span>“How does this work?”</span>
                  <span>
                    <LogoMark />
                    Let’s take it step by step.
                  </span>
                </div>
                <span className="step-counter">03</span>
                <h3>Follow your curiosity.</h3>
                <p>
                  Ask a specific question, switch study modes, and explore the
                  source passages behind an answer.
                </p>
                <Link href="/app">
                  Ask your first question
                  <ArrowUpRight size={13} />
                </Link>
              </article>
            </div>
          </div>
        </section>
        <section
          className="workspace-section site-width section-space"
          id="your-space"
          data-reveal
        >
          <div className="section-intro">
            <div>
              <span className="section-kicker">A place to come back to.</span>
              <h2>
                Keep your learning
                <br />
                close at hand.
              </h2>
            </div>
            <p>
              A quiet workspace that remembers the useful things, and gives the
              next question room to grow.
            </p>
          </div>
          <div className="workspace-features">
            <article className="feature-saved">
              <div className="saved-visual">
                <span>
                  <Bookmark size={14} fill="currentColor" />
                  Saved for later
                </span>
                <div>
                  <BookOpen size={17} />
                  <span>
                    Ohm’s law, in simple steps
                    <small>A useful explanation to revisit.</small>
                  </span>
                  <Check size={14} />
                </div>
                <div>
                  <Layers size={17} />
                  <span>
                    A little electricity revision
                    <small>The important ideas, together.</small>
                  </span>
                  <Check size={14} />
                </div>
              </div>
              <h3>Save your moments of clarity.</h3>
              <p>
                Bookmark an explanation, find a past session, or export a
                conversation as Markdown. Keep the ideas that help.
              </p>
            </article>
            <article className="feature-theme">
              <div className="theme-visual">
                <span className="theme-preview-light">
                  <span />
                  <span />
                  <span />
                  <span />
                </span>
                <span className="theme-preview-dark">
                  <span />
                  <span />
                  <span />
                  <span />
                </span>
                <span className="theme-mini-palette">
                  <i />
                  <i />
                  <i />
                  <i />
                </span>
              </div>
              <h3>Find your kind of focus.</h3>
              <p>
                Light, dark, or in step with your device. Choose a color palette
                and fonts to make the whole space feel familiar.
              </p>
            </article>
            <article className="feature-sources">
              <div className="source-visual">
                <div>
                  <span>1</span>Sample science notes<small>PDF page 1</small>
                </div>
                <p>
                  “At constant temperature, voltage equals current times
                  resistance.”
                </p>
                <span className="source-underline" />
              </div>
              <h3>Stay close to the source.</h3>
              <p>
                Open the passages used for an answer. Page references help you
                return to the original and check the explanation.
              </p>
            </article>
          </div>
        </section>
        <section className="privacy-section site-width" data-reveal>
          <div className="privacy-emblem">
            <ShieldCheck size={38} strokeWidth={1.2} />
          </div>
          <div>
            <span className="section-kicker">
              A thoughtful space for your material.
            </span>
            <h2>Know what goes where.</h2>
            <p>
              Your PDFs are indexed on the app’s server. Your key stays out of
              the database. When you ask a question, relevant excerpts and
              recent conversation context go to NVIDIA to prepare the answer.
            </p>
            <Link href="/privacy">
              A clear look at your privacy
              <ArrowUpRight size={14} />
            </Link>
          </div>
          <div className="privacy-checks">
            <span>
              <Check size={14} />
              No local model download
            </span>
            <span>
              <Check size={14} />
              Your own NVIDIA API key
            </span>
            <span>
              <Check size={14} />
              Delete books from your library
            </span>
          </div>
        </section>
        <section
          className="faq-section site-width section-space"
          id="questions"
          data-reveal
        >
          <div className="faq-intro">
            <span className="section-kicker">Before you turn the page.</span>
            <h2>
              A few good
              <br />
              questions.
            </h2>
            <p>And a little clarity on how it all works.</p>
            <span className="faq-doodle">
              <LogoMark />
            </span>
          </div>
          <div className="faq-list">
            {[
              [
                "What do I need to get started?",
                `A text-based textbook PDF and an NVIDIA NIM API key. ${guestAllowed ? "You can open the guest workspace without an account." : "Sign in to open your study workspace."} Sample science notes are included if you want to explore first.`,
              ],
              [
                "Can I use it without paying for a subscription?",
                "This project has no app subscription. You bring your own NVIDIA key. NVIDIA’s developer model access, availability, and request allowances depend on your account; check the API Catalog for current terms.",
              ],
              [
                "Are the answers always correct?",
                "AI answers can contain mistakes, even when they include citations. Use the page references and source excerpts to check an explanation against your textbook. Broad revision prompts work from selected passages, not an exhaustive reading of every page.",
              ],
              [
                "Which PDFs work best?",
                "Text-based PDFs with selectable text work best. You can add up to 20 books, with a limit of 20 MB and 700 pages per PDF. A focused chapter is often easier to study. Image-only scans need text recognition first.",
              ],
              [
                "What does signing in give me?",
                "Signing in links the textbook library on this app’s server to your provider identity. Guest and signed-in workspaces are separate. Chat history and appearance choices are stored on the device you use; you can export conversations to keep a copy.",
              ],
              [
                "Can I make the space feel more like me?",
                "Yes. Open Settings, choose Appearance, and select light, dark, or system mode. There are six accent palettes, four body fonts, and three heading styles. Your choices apply across the landing page, sign-in pages, and study workspace.",
              ],
            ].map(([question, answer]) => (
              <details key={question}>
                <summary>
                  {question}
                  <ChevronDown size={17} />
                </summary>
                <p>{answer}</p>
              </details>
            ))}
          </div>
        </section>
        <section className="landing-cta site-width" data-reveal>
          <div className="cta-decoration" aria-hidden="true">
            <LogoMark />
          </div>
          <span className="section-kicker">A fresh page is waiting.</span>
          <h2>
            Bring a question.
            <br />
            Find a little clarity.
          </h2>
          <p>Make your next study session a little more your own.</p>
          <Link href="/app" className="hero-primary">
            Open your study space
            <ArrowUpRight size={17} />
          </Link>
          <small>Your books. Your key. Your pace.</small>
        </section>
      </main>
      <footer className="site-footer site-width">
        <div>
          <Link href="/" aria-label="NCERT Tutor home">
            <Logo />
          </Link>
          <p>A little help. A lot of learning.</p>
        </div>
        <div className="footer-links">
          <a href="#ways-to-learn">Ways to learn</a>
          <Link href="/privacy">Data & privacy</Link>
          <Link href="/sign-in">Sign in</Link>
          <a
            href="https://github.com/jeevanu345/NCERT-RAG-Bot"
            target="_blank"
            rel="noreferrer"
          >
            View the project
            <ArrowUpRight size={11} />
          </a>
        </div>
        <p className="footer-note">
          An independent study companion. Not affiliated with NCERT. Check AI
          answers against your textbook.
        </p>
      </footer>
    </div>
  );
}
function ModePreview({
  mode,
  onModeChange,
}: {
  mode: DemoMode;
  onModeChange: (mode: DemoMode) => void;
}) {
  // Each example is clearly labelled and uses the included original notes.
  return (
    <div className={`mode-preview mode-preview-${mode}`}>
      <span className="mode-preview-note">
        <Sparkles size={13} />A small example of{" "}
        {modes.find((m) => m.id === mode)?.label.toLowerCase()}
      </span>
      <ProductDemo
        key={mode}
        large
        initialMode={mode}
        onModeChange={onModeChange}
      />
      <p className="mode-preview-caption">
        Try all four modes in the preview above, then bring your own chapter.
      </p>
    </div>
  );
}
