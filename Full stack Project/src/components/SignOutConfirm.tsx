"use client";
import { useState } from "react";
import Link from "next/link";
import { LoaderCircle, LogOut } from "lucide-react";
import { finishSignOut } from "@/app/auth/actions";
export default function SignOutConfirm({ accountId }: { accountId: string }) {
  const [clear, setClear] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      // Cookie changes refresh Server Components and can unmount this form.
      // Complete the user's chosen local cleanup before the server redirects.
      try {
        sessionStorage.removeItem(`ncert-key-v1:${accountId}`);
        sessionStorage.removeItem("ncert-key-v1");
        if (clear) localStorage.removeItem(`ncert-sessions-v1:${accountId}`);
        localStorage.removeItem("ncert-sessions-v1");
      } catch {
        /* Blocked browser storage must not prevent cookie sign-out. */
      }
      await finishSignOut();
    } catch {
      setError("We could not finish signing out. Please try again.");
      setBusy(false);
    }
  }
  return (
    <form onSubmit={submit} className="signout-form">
      <label className="check-label">
        <input
          type="checkbox"
          checked={clear}
          disabled={busy}
          onChange={(e) => setClear(e.target.checked)}
        />
        Remove this account’s saved chats from this device
      </label>
      <p>
        Your textbook library will be here when you return. Your remembered API
        key will be removed from this tab.
      </p>
      {error && (
        <p role="alert" className="inline-error">
          {error}
        </p>
      )}
      <div className="signout-actions">
        <Link href="/app" className="secondary-link">
          Keep studying
        </Link>
        <button className="primary-link" disabled={busy} type="submit">
          {busy ? (
            <LoaderCircle size={15} className="spin" />
          ) : (
            <LogOut size={15} />
          )}
          Sign out
        </button>
      </div>
    </form>
  );
}
