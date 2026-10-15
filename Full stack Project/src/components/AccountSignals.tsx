"use client";
import { useEffect } from "react";
export default function AccountSignals({ accountId }: { accountId: string }) {
  useEffect(() => {
    if (!("BroadcastChannel" in window)) return;
    const channel = new BroadcastChannel("ncert-account");
    channel.onmessage = (event) => {
      if (
        event.data === "signed-out" ||
        (event.data?.type === "identity" &&
          typeof event.data.id === "string" &&
          event.data.id !== accountId)
      ) {
        try {
          for (const key of Object.keys(sessionStorage))
            if (key.startsWith("ncert-key-v1")) sessionStorage.removeItem(key);
        } catch {}
        window.location.reload();
      }
    };
    channel.postMessage({ type: "identity", id: accountId });
    return () => channel.close();
  }, [accountId]);
  return null;
}
