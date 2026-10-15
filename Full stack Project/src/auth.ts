import NextAuth, { AuthError } from "next-auth";
import Google from "next-auth/providers/google";
import MicrosoftEntraID from "next-auth/providers/microsoft-entra-id";
import Apple from "next-auth/providers/apple";
import type { Provider } from "next-auth/providers";
import { createHash } from "node:crypto";
import { cache } from "react";
import {
  authConfigured,
  providerAvailability,
  type AccountInfo,
} from "@/lib/auth-config";

function workspaceForAccount(provider: string, subject: string) {
  // A signed provider identity, never an email or an untrusted browser header.
  const hex = createHash("sha256")
    .update(`ncert-user:${provider}:${subject}`)
    .digest("hex");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-4${hex.slice(13, 16)}-a${hex.slice(17, 20)}-${hex.slice(20, 32)}`;
}
const providers: Provider[] = [];
const enabled = new Set(
  providerAvailability()
    .filter((provider) => provider.enabled)
    .map((provider) => provider.id),
);
if (enabled.has("google"))
  providers.push(
    Google({
      clientId: process.env.AUTH_GOOGLE_ID,
      clientSecret: process.env.AUTH_GOOGLE_SECRET,
      authorization: { params: { scope: "openid email profile" } },
    }),
  );
if (enabled.has("microsoft-entra-id"))
  providers.push(
    MicrosoftEntraID({
      clientId: process.env.AUTH_MICROSOFT_ENTRA_ID_ID,
      clientSecret: process.env.AUTH_MICROSOFT_ENTRA_ID_SECRET,
      issuer:
        process.env.AUTH_MICROSOFT_ENTRA_ID_ISSUER ||
        "https://login.microsoftonline.com/common/v2.0",
    }),
  );
if (enabled.has("apple"))
  providers.push(
    Apple({
      clientId: process.env.AUTH_APPLE_ID,
      clientSecret: process.env.AUTH_APPLE_SECRET,
    }),
  );
const secure = process.env.AUTH_URL?.startsWith("https://") ?? false;
// Apple sends a cross-site POST. Only the short-lived OAuth check cookies need
// SameSite=None over HTTPS; the encrypted session retains Auth.js defaults.
const appleCookies =
  secure && process.env.AUTH_APPLE_ID
    ? {
        state: {
          name: "__Secure-authjs.state",
          options: {
            httpOnly: true,
            sameSite: "none" as const,
            secure: true,
            path: "/",
            maxAge: 900,
          },
        },
        nonce: {
          name: "__Secure-authjs.nonce",
          options: {
            httpOnly: true,
            sameSite: "none" as const,
            secure: true,
            path: "/",
            maxAge: 900,
          },
        },
        pkceCodeVerifier: {
          name: "__Secure-authjs.pkce.code_verifier",
          options: {
            httpOnly: true,
            sameSite: "none" as const,
            secure: true,
            path: "/",
            maxAge: 900,
          },
        },
      }
    : undefined;
export const { handlers, auth, signIn, signOut } = NextAuth({
  providers,
  secret: process.env.AUTH_SECRET,
  trustHost: true,
  debug: false,
  logger: {
    // Log a useful error code without provider responses, tokens, or profile data.
    error(error) {
      console.error(`[auth] ${error instanceof AuthError ? error.type : 'AuthenticationError'}`);
    },
  },
  session: { strategy: "jwt", maxAge: 60 * 60 * 24 * 7 },
  pages: { signIn: "/sign-in", signOut: "/sign-out", error: "/auth/error" },
  cookies: appleCookies,
  callbacks: {
    jwt({ token, account }) {
      if (account?.providerAccountId)
        token.workspaceId = workspaceForAccount(
          account.provider,
          account.providerAccountId,
        );
      // Provider access/refresh tokens are deliberately never exposed to the browser.
      return token;
    },
    session({ session, token }) {
      if (session.user && typeof token.workspaceId === "string")
        session.user.id = token.workspaceId;
      return session;
    },
    redirect({ url, baseUrl }) {
      const destination = new URL(url, baseUrl);
      return destination.origin === new URL(baseUrl).origin
        ? destination.toString()
        : `${baseUrl}/app`;
    },
  },
});
export const currentAccount = cache(
  async function currentAccount(): Promise<AccountInfo | null> {
    if (!authConfigured()) return null;
    const session = await auth();
    if (!session?.user?.id) return null;
    return {
      id: session.user.id,
      name: session.user.name || "Student",
      email: session.user.email || null,
    };
  },
);
