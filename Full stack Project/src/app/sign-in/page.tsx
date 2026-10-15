import Link from "next/link";
import { redirect } from "next/navigation";
import { ArrowLeft, BookOpen, Check, ShieldCheck } from "lucide-react";
import Logo from "@/components/brand/Logo";
import ThemeToggle from "@/components/ThemeToggle";
import AuthSubmit from "@/components/AuthSubmit";
import { currentAccount } from "@/auth";
import {
  AUTH_ERROR_MESSAGES,
  authenticationRequired,
  providerAvailability,
  safeReturnTo,
} from "@/lib/auth-config";
import { startSignIn } from "@/app/auth/actions";
export const dynamic = "force-dynamic";
export const metadata = {
  title: "Sign in",
  robots: { index: false, follow: false },
};
export default async function SignInPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const returnTo = safeReturnTo(params.returnTo);
  if (await currentAccount()) redirect(returnTo);
  const providers = providerAvailability();
  const available = providers.some((p) => p.enabled);
  const error =
    typeof params.error === "string"
      ? AUTH_ERROR_MESSAGES[params.error] || AUTH_ERROR_MESSAGES.default
      : "";
  return (
    <main className="auth-page">
      <aside className="auth-story">
        <Link href="/" aria-label="NCERT Tutor home">
          <Logo />
        </Link>
        <div className="auth-story-content">
          <span className="auth-book">
            <BookOpen size={42} strokeWidth={1.2} />
          </span>
          <h2>
            A familiar chapter.
            <br />A fresh way to see it.
          </h2>
          <p>
            Bring your questions, your textbooks, and your own pace. We’ll make
            a little room for understanding.
          </p>
          <div className="auth-benefits">
            <span>
              <Check size={15} />
              Your own textbook library
            </span>
            <span>
              <Check size={15} />
              Explanations you can trace
            </span>
            <span>
              <Check size={15} />
              Four ways to make it stick
            </span>
          </div>
        </div>
        <p className="auth-story-foot">
          Good questions are always welcome here.
        </p>
      </aside>
      <section className="auth-form-panel">
        <div className="auth-topline">
          <Link href="/">
            <ArrowLeft size={15} />
            Back home
          </Link>
          <Link
            className="auth-mobile-brand"
            href="/"
            aria-label="NCERT Tutor home"
          >
            <Logo compact />
          </Link>
          <ThemeToggle />
        </div>
        <div className="auth-form-content">
          <span className="auth-small-label">Your study space</span>
          <h1>
            Welcome to your
            <br />
            next chapter.
          </h1>
          <p>Sign in to keep your textbook library connected to you.</p>
          {params.signedOut === "1" && (
            <div className="auth-success" role="status">
              <Check size={16} />
              You’re signed out. See you next chapter.
            </div>
          )}
          {error && (
            <div className="auth-error" role="alert">
              {error}
            </div>
          )}
          <div className="oauth-options">
            {providers.map((p) => (
              <form action={startSignIn} key={p.id}>
                <input type="hidden" name="provider" value={p.id} />
                <input type="hidden" name="returnTo" value={returnTo} />
                <AuthSubmit
                  provider={p.id}
                  label={p.label}
                  enabled={p.enabled}
                />
              </form>
            ))}
          </div>
          {providers.some((p) => !p.enabled) && (
            <p id="provider-availability" className="provider-availability">
              {available
                ? "Some sign-in options are not available yet."
                : "Account sign-in is being configured."}{" "}
              {authenticationRequired()
                ? "Please check back shortly."
                : "You can still use the study space as a guest."}
            </p>
          )}
          {!authenticationRequired() && (
            <>
              <div className="auth-or">
                <span />
                or
                <span />
              </div>
              <Link className="guest-button" href="/app">
                Continue as a guest<span>Stay in this browser</span>
              </Link>
            </>
          )}
          <p className="auth-privacy">
            <ShieldCheck size={15} />
            <span>
              Your NVIDIA key is separate from your sign-in.
              <br />
              <Link href="/privacy">How your information is handled</Link>
            </span>
          </p>
        </div>
        <p className="auth-bottom">A little help. A lot of learning.</p>
      </section>
    </main>
  );
}
