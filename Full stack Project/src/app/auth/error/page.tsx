import Link from "next/link";
import { CircleHelp } from "lucide-react";
import Logo from "@/components/brand/Logo";
import { AUTH_ERROR_MESSAGES, authenticationRequired } from "@/lib/auth-config";
export const metadata = {
  title: "Sign-in help",
  robots: { index: false, follow: false },
};
export default async function AuthErrorPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const { error } = await searchParams;
  const message =
    AUTH_ERROR_MESSAGES[typeof error === "string" ? error : "default"] ||
    AUTH_ERROR_MESSAGES.default;
  return (
    <main className="account-page">
      <header>
        <Link href="/" aria-label="NCERT Tutor home">
          <Logo />
        </Link>
      </header>
      <section className="account-panel">
        <span className="page-icon">
          <CircleHelp size={25} />
        </span>
        <h1>Let’s try that again.</h1>
        <p>{message}</p>
        <div className="signout-actions">
          {!authenticationRequired() && (
            <Link className="secondary-link" href="/app">
              Continue as a guest
            </Link>
          )}
          <Link className="primary-link" href="/sign-in">
            Back to sign in
          </Link>
        </div>
      </section>
    </main>
  );
}
