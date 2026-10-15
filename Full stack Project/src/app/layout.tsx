import type { Metadata } from "next";
import "@fontsource-variable/manrope";
import "@fontsource-variable/inter";
import "@fontsource-variable/nunito-sans";
import "@fontsource-variable/lora";
import "./globals.css";
import "./appearance.css";
import "./landing.css";
import AppearanceProvider from "@/components/AppearanceProvider";
import AccountSignals from "@/components/AccountSignals";
import { appearanceBootScript } from "@/lib/appearance";
import { currentAccount } from "@/auth";
// Account controls must reflect the current session, including after an OAuth callback.
export const dynamic = "force-dynamic";
export const metadata: Metadata = {
  title: {
    default: "NCERT Tutor · A clearer next chapter",
    template: "%s · NCERT Tutor",
  },
  description:
    "Your NCERT textbooks, a little curiosity, and a clearer way to learn. Explore source-backed explanations, revision notes, practice questions, and flashcards with your NVIDIA NIM key.",
  icons: { icon: "/brand/icon.svg" },
  openGraph: {
    title: "NCERT Tutor · A clearer next chapter",
    description:
      "Your own textbooks. Four ways to learn. A study space made for your pace.",
    type: "website",
    siteName: "NCERT Tutor",
  },
};
export default async function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const account = await currentAccount();
  return (
    <html
      lang="en"
      suppressHydrationWarning
      data-palette="forest"
      data-font="manrope"
      data-heading="editorial"
    >
      <head>
        <script
          id="ncert-appearance"
          dangerouslySetInnerHTML={{ __html: appearanceBootScript }}
        />
      </head>
      <body>
        <AppearanceProvider>
          <AccountSignals accountId={account?.id || "guest"} />
          {children}
        </AppearanceProvider>
      </body>
    </html>
  );
}
