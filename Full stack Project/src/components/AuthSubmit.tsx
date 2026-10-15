"use client";
import { useFormStatus } from "react-dom";
import { LoaderCircle } from "lucide-react";
import ProviderIcon from "./ProviderIcon";
export default function AuthSubmit({
  provider,
  label,
  enabled,
}: {
  provider: string;
  label: string;
  enabled: boolean;
}) {
  const { pending } = useFormStatus();
  return (
    <button
      className="oauth-button"
      disabled={!enabled || pending}
      type="submit"
      aria-describedby={!enabled ? "provider-availability" : undefined}
    >
      {pending ? (
        <LoaderCircle size={18} className="spin" />
      ) : (
        <ProviderIcon provider={provider} />
      )}
      <span>{pending ? "Connecting…" : `Continue with ${label}`}</span>
      {!enabled && <small>Unavailable</small>}
    </button>
  );
}
