"use client";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
export default function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  return (
    <button
      className="theme-toggle"
      aria-label="Toggle light and dark mode"
      title="Toggle light and dark mode"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
    >
      <Sun className="light-mode-icon" size={18} />
      <Moon className="dark-mode-icon" size={18} />
    </button>
  );
}
