import Link from "next/link";
import { currentAccount } from "@/auth";
import Logo from "@/components/brand/Logo";
import SignOutConfirm from "@/components/SignOutConfirm";
import ThemeToggle from "@/components/ThemeToggle";
export const dynamic = "force-dynamic";
export const metadata = {
  title: "Sign out",
  robots: { index: false, follow: false },
};
export default async function SignOutPage() {
  const account = await currentAccount();
  return (
    <main className="account-page">
      <header>
        <Link href="/" aria-label="NCERT Tutor home">
          <Logo />
        </Link>
        <ThemeToggle />
      </header>
      <section className="account-panel">
        <span className="account-page-mark">
          <Logo compact />
        </span>
        <h1>
          {account ? "Pause here. Pick up later." : "You’re not signed in."}
        </h1>
        <p>
          {account
            ? `Ready to sign out${account.name ? `, ${account.name.split(" ")[0]}` : ""}?`
            : "You can sign in whenever you’re ready to continue."}
        </p>
        {account ? (
          <SignOutConfirm accountId={account.id} />
        ) : (
          <div className="signout-actions">
            <Link href="/" className="secondary-link">
              Back home
            </Link>
            <Link href="/sign-in" className="primary-link">
              Sign in
            </Link>
          </div>
        )}
      </section>
    </main>
  );
}
