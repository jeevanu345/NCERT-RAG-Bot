"use client";

import { useState } from "react";
import { Tabs } from "radix-ui";
import {
  ArrowUpRight,
  Check,
  Eye,
  EyeOff,
  KeyRound,
  Palette,
  LoaderCircle,
  ShieldCheck,
  Unplug,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DEFAULT_MODEL, readResponse } from "@/lib/types";
import Modal from "./Modal";
import AppearanceSettings from "./AppearanceSettings";

export type Connection = {
  key: string;
  model: string;
  connected: boolean;
  remember: boolean;
};

export default function ConnectionSettings({
  open,
  onOpenChange,
  connection,
  serverKey,
  identity,
  onSave,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  connection: Connection;
  serverKey: boolean;
  identity: string;
  onSave: (connection: Connection) => void;
}) {
  const [tab, setTab] = useState("connection");
  const [key, setKey] = useState(connection.key);
  const [model, setModel] = useState(connection.model);
  const [remember, setRemember] = useState(connection.remember);
  const [visible, setVisible] = useState(false);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  async function connect(event: React.FormEvent) {
    event.preventDefault();
    setError("");
    setSuccess(false);
    const cleanKey = key.trim();
    if (!cleanKey && !serverKey) {
      setError("Paste your NVIDIA API key to connect.");
      return;
    }
    if (cleanKey && (!cleanKey.startsWith("nvapi-") || /\s/.test(cleanKey))) {
      setError(
        "Your NVIDIA API key should start with nvapi- and contain no spaces.",
      );
      return;
    }
    if (!/^[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+$/.test(model.trim())) {
      setError(
        "Use the model ID from the NVIDIA catalog, such as meta/llama-3.3-70b-instruct.",
      );
      return;
    }
    setTesting(true);
    try {
      await readResponse(
        await fetch("/api/connection", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-NCERT-Identity": identity,
            ...(cleanKey ? { Authorization: `Bearer ${cleanKey}` } : {}),
          },
          body: JSON.stringify({ model: model.trim() }),
          signal: AbortSignal.timeout(115000),
        }),
      );
      onSave({ key: cleanKey, model: model.trim(), connected: true, remember });
      setSuccess(true);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not connect. Please retry.",
      );
    } finally {
      setTesting(false);
    }
  }

  return (
    <Modal
      open={open}
      onOpenChange={(value) => {
        if (!testing) onOpenChange(value);
      }}
      title="Settings"
      description="Your AI connection and a space that feels like you."
    >
      <Tabs.Root value={tab} onValueChange={setTab}>
        <Tabs.List className="settings-tabs" aria-label="Settings sections">
          <Tabs.Trigger value="connection">
            <KeyRound size={14} />
            AI connection
          </Tabs.Trigger>
          <Tabs.Trigger value="appearance">
            <Palette size={14} />
            Appearance
          </Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="appearance">
          <AppearanceSettings />
        </Tabs.Content>
        <Tabs.Content value="connection">
          <div className="provider-card">
            <div className="provider-logo">N</div>
            <div>
              <strong>NVIDIA NIM</strong>
              <p>Hosted models · Your API key</p>
            </div>
            <span className="pill">BYOK</span>
          </div>
          <form onSubmit={connect} className="settings-form">
            <div className="field-heading">
              <label htmlFor="nim-key">API key</label>
              <a
                href="https://build.nvidia.com/meta/llama-3_3-70b-instruct"
                target="_blank"
                rel="noreferrer"
              >
                Get a key <ArrowUpRight size={13} />
              </a>
            </div>
            <div className="password-field">
              <KeyRound size={16} />
              <Input
                id="nim-key"
                type={visible ? "text" : "password"}
                autoComplete="off"
                spellCheck={false}
                placeholder={
                  serverKey
                    ? "Server key available, or paste your own"
                    : "nvapi-…"
                }
                value={key}
                disabled={testing}
                maxLength={512}
                onChange={(e) => {
                  setKey(e.target.value);
                  setSuccess(false);
                }}
              />
              <Button
                variant="ghost"
                size="icon"
                type="button"
                aria-label={visible ? "Hide API key" : "Show API key"}
                onClick={() => setVisible(!visible)}
              >
                {visible ? <EyeOff /> : <Eye />}
              </Button>
            </div>
            <p className="field-help">
              Your key is sent only to this app’s backend and NVIDIA. It is
              never saved to the database.
            </p>
            <label htmlFor="nim-model">Model</label>
            <select
              id="model-preset"
              aria-label="Choose a model preset"
              disabled={testing}
              value={
                [DEFAULT_MODEL, "meta/llama-3.1-8b-instruct"].includes(model)
                  ? model
                  : "custom"
              }
              onChange={(e) => {
                setModel(e.target.value === "custom" ? "" : e.target.value);
                setSuccess(false);
              }}
            >
              <option value={DEFAULT_MODEL}>Llama 3.3 70B · Balanced</option>
              <option value="meta/llama-3.1-8b-instruct">
                Llama 3.1 8B · Lightweight
              </option>
              <option value="custom">Use another NVIDIA model</option>
            </select>
            <Input
              id="nim-model"
              aria-label="NVIDIA model ID"
              value={model}
              disabled={testing}
              maxLength={160}
              onChange={(e) => {
                setModel(e.target.value);
                setSuccess(false);
              }}
              placeholder="publisher/model-name"
            />
            <p className="field-help">
              NVIDIA offers developer access for prototyping. Availability and
              request limits depend on your account.
            </p>
            <label className="check-label">
              <input
                type="checkbox"
                checked={remember}
                disabled={testing}
                onChange={(e) => setRemember(e.target.checked)}
              />
              Remember my key for this tab
            </label>
            <div className="privacy-note">
              <ShieldCheck size={17} />
              <p>
                Textbook text is indexed on this app’s server. Relevant excerpts
                and your conversation are sent to NVIDIA when you ask a
                question.
              </p>
            </div>
            {error && (
              <p className="inline-error" role="alert">
                {error}
              </p>
            )}
            {success && (
              <p className="inline-success" role="status">
                <Check size={16} />
                Connected. You’re ready to study.
              </p>
            )}
            <div className="modal-actions">
              {connection.connected && (
                <Button
                  variant="ghost"
                  type="button"
                  disabled={testing}
                  onClick={() => {
                    onSave({
                      key: "",
                      model,
                      connected: false,
                      remember: false,
                    });
                    setKey("");
                    setSuccess(false);
                  }}
                >
                  <Unplug size={15} />
                  Disconnect
                </Button>
              )}
              <Button type="submit" disabled={testing}>
                {testing ? <LoaderCircle className="spin" /> : <Check />}
                {testing ? "Testing connection…" : "Test & save connection"}
              </Button>
            </div>
          </form>
        </Tabs.Content>
      </Tabs.Root>
    </Modal>
  );
}
