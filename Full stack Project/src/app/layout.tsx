import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NCERT Tutor — Your space to understand",
  description:
    "A focused Class 10 study companion. Bring your NVIDIA NIM key, upload a textbook, and learn with page-referenced answers.",
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
