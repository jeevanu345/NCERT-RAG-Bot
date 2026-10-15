"use client";
import { createContext, useContext, useEffect, useState } from "react";
import { ThemeProvider } from "next-themes";
import {
  APPEARANCE_KEY,
  DEFAULT_APPEARANCE,
  parseAppearance,
  type Appearance,
} from "@/lib/appearance";
const Context = createContext<{
  appearance: Appearance;
  update: (next: Partial<Appearance>) => void;
  reset: () => void;
  storageError: string;
}>({
  appearance: DEFAULT_APPEARANCE,
  update: () => {},
  reset: () => {},
  storageError: "",
});
function apply(next: Appearance) {
  const root = document.documentElement;
  root.dataset.palette = next.palette;
  root.dataset.font = next.font;
  root.dataset.heading = next.heading;
}
export default function AppearanceProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [appearance, setAppearance] = useState<Appearance>(DEFAULT_APPEARANCE);
  const [storageError, setStorageError] = useState("");
  useEffect(() => {
    let alive = true;
    const sync = () => {
      if (!alive) return;
      try {
        const next = parseAppearance(
          JSON.parse(localStorage.getItem(APPEARANCE_KEY) || "{}"),
        );
        apply(next);
        setAppearance(next);
      } catch {
        setStorageError(
          "Your appearance works for this visit. This browser could not save it.",
        );
      }
    };
    queueMicrotask(sync);
    window.addEventListener("storage", sync);
    return () => {
      alive = false;
      window.removeEventListener("storage", sync);
    };
  }, []);
  function update(next: Partial<Appearance>) {
    const value = parseAppearance({ ...appearance, ...next });
    apply(value);
    setAppearance(value);
    try {
      localStorage.setItem(APPEARANCE_KEY, JSON.stringify(value));
      setStorageError("");
    } catch {
      setStorageError(
        "Your appearance works for this visit. This browser could not save it.",
      );
    }
  }
  return (
    <ThemeProvider
      attribute="data-theme"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
      storageKey="ncert-color-mode"
    >
      <Context.Provider
        value={{
          appearance,
          update,
          reset: () => update(DEFAULT_APPEARANCE),
          storageError,
        }}
      >
        {children}
      </Context.Provider>
    </ThemeProvider>
  );
}
export const useAppearance = () => useContext(Context);
