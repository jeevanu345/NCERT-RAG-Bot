import Link from "next/link";
import Logo from "@/components/brand/Logo";
import ThemeToggle from "@/components/ThemeToggle";
export const metadata = { title: "Data & privacy" };
export default function PrivacyPage() {
  return (
    <main className="privacy-page site-width">
      <header>
        <Link href="/" aria-label="NCERT Tutor home">
          <Logo />
        </Link>
        <ThemeToggle />
      </header>
      <article>
        <span className="section-kicker">
          A clear look at your information.
        </span>
        <h1>
          Your study material,
          <br />
          and where it goes.
        </h1>
        <p className="privacy-lead">
          This page describes how this version of NCERT Tutor handles your data.
          The app is designed to run on your own computer; a hosted operator is
          responsible for their deployment and its data practices.
        </p>
        <section>
          <h2>Your textbook library</h2>
          <p>
            Uploaded PDF text is extracted and stored in the app’s SQLite
            database with document names, page numbers, and a document
            fingerprint. Original PDF bytes are discarded after extraction. The
            default database stays on the computer running the Python service.
            Removing a textbook deletes its passages and search entries.
          </p>
        </section>
        <section>
          <h2>Your NVIDIA key and questions</h2>
          <p>
            Your NVIDIA NIM key stays in page memory by default. Remembering it
            for a tab uses browser session storage. Keys are not saved to the
            textbook database or chat history by the app. A connection test
            sends a small request to NVIDIA. Asking a question sends your key,
            recent conversation context, and relevant textbook excerpts to
            NVIDIA’s hosted API.
          </p>
          <p>
            NVIDIA handles requests under its own{" "}
            <a
              href="https://www.nvidia.com/en-us/about-nvidia/privacy-policy/"
              target="_blank"
              rel="noreferrer"
            >
              privacy policy
            </a>{" "}
            and service terms. A local search that finds no relevant passages
            can return a clarification without a model request.
          </p>
        </section>
        <section>
          <h2>Signing in</h2>
          <p>
            Google, Microsoft, and Apple OAuth are supported when the operator
            configures the provider credentials. Authentication uses Auth.js and
            an encrypted, HTTP-only session cookie lasting up to seven days. The
            app uses the verified provider identity to select your library.
            Provider access tokens are not exposed to the browser.
          </p>
          <p>
            Different providers are separate identities; the app does not merge
            accounts just because they use the same email address. Guest study
            uses a separate browser workspace cookie. Signing in does not
            automatically import a guest library.
          </p>
        </section>
        <section>
          <h2>Saved on your device</h2>
          <p>
            Recent chats, saved answers, and appearance choices use local
            storage. Remembered NVIDIA keys use tab storage. Signed-in chat and
            key storage are separated by account. Chat history is not
            automatically synced between devices. You can delete sessions or
            export a Markdown copy. Up to 20 recent sessions and 40 messages per
            session are retained.
          </p>
        </section>
        <section>
          <h2>Signing out and deletion</h2>
          <p>
            Signing out revokes the current browser session and removes
            remembered keys from the tab. You can also remove this account’s
            saved chats from the device on the sign-out page. Your signed-in
            library stays on the server until its textbooks are removed. Remove
            books before signing out if you want their extracted text deleted.
            Clearing a guest workspace cookie can leave its old library
            inaccessible to that browser; the local operator can remove unused
            local data.
          </p>
        </section>
        <section>
          <h2>Fonts, analytics, and limitations</h2>
          <p>
            Fonts are self-hosted. This version adds no advertising or analytics
            trackers. OAuth providers receive authentication requests when you
            choose to sign in. AI answers can be inaccurate. Check the cited
            passages against your original textbook.
          </p>
        </section>
        <Link className="primary-link" href="/app">
          Back to your study space
        </Link>
      </article>
    </main>
  );
}
