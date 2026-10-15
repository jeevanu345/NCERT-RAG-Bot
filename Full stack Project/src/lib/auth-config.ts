export const OAUTH_PROVIDERS = [
  {
    id: "google",
    label: "Google",
    idEnv: "AUTH_GOOGLE_ID",
    secretEnv: "AUTH_GOOGLE_SECRET",
  },
  {
    id: "microsoft-entra-id",
    label: "Microsoft",
    idEnv: "AUTH_MICROSOFT_ENTRA_ID_ID",
    secretEnv: "AUTH_MICROSOFT_ENTRA_ID_SECRET",
  },
  {
    id: "apple",
    label: "Apple",
    idEnv: "AUTH_APPLE_ID",
    secretEnv: "AUTH_APPLE_SECRET",
  },
] as const;
export type ProviderId = (typeof OAUTH_PROVIDERS)[number]["id"];
export type AccountInfo = { id: string; name: string; email: string | null };
export function providerAvailability(
  env: Record<string, string | undefined> = process.env,
) {
  const secretReady = (env.AUTH_SECRET?.trim().length ?? 0) >= 32;
  return OAUTH_PROVIDERS.map((p) => ({
    id: p.id,
    label: p.label,
    enabled:
      secretReady &&
      !!env[p.idEnv]?.trim() &&
      !!env[p.secretEnv]?.trim() &&
      (p.id !== "apple" || !!env.AUTH_URL?.startsWith("https://")),
  }));
}
export function authConfigured() {
  return providerAvailability().some((p) => p.enabled);
}
export function authenticationRequired() {
  return process.env.AUTH_REQUIRED === "true";
}
export function safeReturnTo(value: unknown) {
  // Only our study route is an OAuth return destination; never redirect to user-provided hosts.
  return typeof value === "string" &&
    /^\/app(?:[/?#]|$)/.test(value) &&
    !value.includes("\\") &&
    !/[\r\n]/.test(value)
    ? value
    : "/app";
}
export const AUTH_ERROR_MESSAGES: Record<string, string> = {
  Configuration: "Sign-in is not available yet. Please try again later.",
  AccessDenied:
    "Sign-in was cancelled or access was declined. You can try again with another account.",
  OAuthAccountNotLinked:
    "This email belongs to another sign-in provider. Use the provider you originally chose.",
  OAuthCallbackError: "We could not finish that sign-in. Please start again.",
  OAuthSignin: "We could not reach your sign-in provider. Please try again.",
  OAuthCallback: "We could not finish that sign-in. Please start again.",
  Verification: "That sign-in request has expired. Please try again.",
  SessionRequired: "Sign in to open your study workspace.",
  default: "We could not complete sign-in. Please try again.",
};
