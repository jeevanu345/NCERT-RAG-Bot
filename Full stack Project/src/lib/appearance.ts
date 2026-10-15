export const PALETTES = [
  { id: "forest", name: "Forest", color: "#216a55" },
  { id: "ocean", name: "Ocean", color: "#23659d" },
  { id: "iris", name: "Iris", color: "#6b52a1" },
  { id: "rose", name: "Rose", color: "#9b4267" },
  { id: "amber", name: "Amber", color: "#805c20" },
  { id: "slate", name: "Slate", color: "#4c6076" },
] as const;
export const FONTS = [
  { id: "manrope", name: "Manrope", description: "Clean & considered" },
  { id: "inter", name: "Inter", description: "Crisp & familiar" },
  { id: "nunito", name: "Nunito Sans", description: "Soft & friendly" },
  { id: "system", name: "System", description: "Your device’s default" },
] as const;
export const HEADINGS = [
  { id: "editorial", name: "Lora", description: "A little character" },
  { id: "modern", name: "Match body", description: "One clear voice" },
  { id: "classic", name: "Georgia", description: "A familiar classic" },
] as const;
export type Appearance = {
  palette: (typeof PALETTES)[number]["id"];
  font: (typeof FONTS)[number]["id"];
  heading: (typeof HEADINGS)[number]["id"];
};
export const DEFAULT_APPEARANCE: Appearance = {
  palette: "forest",
  font: "manrope",
  heading: "editorial",
};
export const APPEARANCE_KEY = "ncert-appearance-v1";
export function parseAppearance(value: unknown): Appearance {
  const data =
    value && typeof value === "object"
      ? (value as Record<string, unknown>)
      : {};
  return {
    palette: PALETTES.some((p) => p.id === data.palette)
      ? (data.palette as Appearance["palette"])
      : "forest",
    font: FONTS.some((f) => f.id === data.font)
      ? (data.font as Appearance["font"])
      : "manrope",
    heading: HEADINGS.some((h) => h.id === data.heading)
      ? (data.heading as Appearance["heading"])
      : "editorial",
  };
}
// Validated attributes only: stored values never become CSS or executable code.
export const appearanceBootScript = `try{const p=JSON.parse(localStorage.getItem('${APPEARANCE_KEY}')||'{}');const d=document.documentElement;for(const [k,v] of Object.entries({palette:['forest','ocean','iris','rose','amber','slate'],font:['manrope','inter','nunito','system'],heading:['editorial','modern','classic']})){d.dataset[k]=v.includes(p[k])?p[k]:v[0]}}catch{}`;
