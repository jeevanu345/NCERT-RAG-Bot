"use client";
import { Check, Monitor, Moon, RotateCcw, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useAppearance } from "./AppearanceProvider";
import { FONTS, HEADINGS, PALETTES, type Appearance } from "@/lib/appearance";
import { Button } from "./ui/button";
export default function AppearanceSettings() {
  const { theme, setTheme } = useTheme();
  const { appearance, update, reset, storageError } = useAppearance();
  return (
    <div className="appearance-settings">
      <div className="settings-section-label">
        <h3>A space that feels like you</h3>
        <p>Colors, type, and a little room to focus.</p>
      </div>
      <fieldset>
        <legend>Color mode</legend>
        <div className="theme-mode-options">
          {[
            { id: "light", label: "Light", icon: Sun },
            { id: "dark", label: "Dark", icon: Moon },
            { id: "system", label: "System", icon: Monitor },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              aria-pressed={(theme || "system") === id}
              onClick={() => setTheme(id)}
            >
              <Icon size={17} />
              {label}
            </button>
          ))}
        </div>
      </fieldset>
      <fieldset>
        <legend>Color palette</legend>
        <div className="palette-options">
          {PALETTES.map((p) => (
            <button
              type="button"
              key={p.id}
              aria-pressed={appearance.palette === p.id}
              onClick={() => update({ palette: p.id })}
            >
              <span className="palette-dot" style={{ background: p.color }}>
                {appearance.palette === p.id && <Check size={14} />}
              </span>
              {p.name}
            </button>
          ))}
        </div>
      </fieldset>
      <div className="font-settings">
        <div>
          <label htmlFor="body-font">Body font</label>
          <select
            id="body-font"
            value={appearance.font}
            onChange={(e) =>
              update({ font: e.target.value as Appearance["font"] })
            }
          >
            {FONTS.map((f) => (
              <option value={f.id} key={f.id}>
                {f.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="heading-font">Heading font</label>
          <select
            id="heading-font"
            value={appearance.heading}
            onChange={(e) =>
              update({ heading: e.target.value as Appearance["heading"] })
            }
          >
            {HEADINGS.map((f) => (
              <option value={f.id} key={f.id}>
                {f.name}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="appearance-preview">
        <span>Your study space, previewed</span>
        <h3>A clearer next chapter.</h3>
        <p>Good questions deserve a little room to grow.</p>
        <span className="preview-pill">
          <Check size={12} />
          Made for your pace
        </span>
      </div>
      <div className="appearance-footer">
        <p>Saved on this device. Applied throughout the app.</p>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          onClick={() => {
            reset();
            setTheme("system");
          }}
        >
          <RotateCcw size={13} />
          Reset
        </Button>
      </div>
      {storageError && (
        <p className="inline-error" role="status">
          {storageError}
        </p>
      )}
    </div>
  );
}
