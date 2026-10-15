export default function ProviderIcon({ provider }: { provider: string }) {
  if (provider === "google")
    return (
      <svg viewBox="0 0 24 24" width="19" height="19" aria-hidden="true">
        <path
          fill="#4285F4"
          d="M21.6 12.2c0-.7-.1-1.4-.2-2.1H12v4h5.4a4.6 4.6 0 0 1-2 3v2.5h3.2c1.9-1.8 3-4.4 3-7.4Z"
        />
        <path
          fill="#34A853"
          d="M12 22c2.7 0 5-.9 6.6-2.4L15.4 17c-.9.6-2 1-3.4 1-2.6 0-4.8-1.8-5.6-4.1H3.1v2.6A10 10 0 0 0 12 22Z"
        />
        <path
          fill="#FBBC05"
          d="M6.4 13.9a6 6 0 0 1 0-3.8V7.5H3.1a10 10 0 0 0 0 9l3.3-2.6Z"
        />
        <path
          fill="#EA4335"
          d="M12 6c1.5 0 2.8.5 3.8 1.5l2.9-2.8A9.6 9.6 0 0 0 12 2a10 10 0 0 0-8.9 5.5l3.3 2.6A6 6 0 0 1 12 6Z"
        />
      </svg>
    );
  if (provider === "microsoft-entra-id")
    return (
      <svg viewBox="0 0 21 21" width="18" height="18" aria-hidden="true">
        <path fill="#f25022" d="M0 0h10v10H0z" />
        <path fill="#7fba00" d="M11 0h10v10H11z" />
        <path fill="#00a4ef" d="M0 11h10v10H0z" />
        <path fill="#ffb900" d="M11 11h10v10H11z" />
      </svg>
    );
  return (
    <svg
      viewBox="0 0 24 24"
      width="21"
      height="21"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M16.4 1.4c.1 1.4-.4 2.8-1.2 3.8-.9 1-2.3 1.8-3.7 1.6-.2-1.4.5-2.8 1.3-3.7.9-1 2.4-1.7 3.6-1.7ZM20.7 17.1c-.5 1.2-.8 1.8-1.5 2.8-1 1.4-2.3 3.2-4 3.2-1.5 0-1.9-1-4-1s-2.5 1-4 1c-1.6 0-2.8-1.6-3.8-3-2.7-4-3-8.6-1.3-11.1a5.4 5.4 0 0 1 4.4-2.6c1.7 0 2.8 1 4.2 1 1.3 0 2.1-1 4.2-1 1.5 0 3.1.8 4.1 2.1-3.6 2-3 7.2 1.7 8.6Z" />
    </svg>
  );
}
