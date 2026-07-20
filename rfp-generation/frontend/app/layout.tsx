import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";
import Providers from "./providers";

export const metadata: Metadata = {
  title: "RFP Response Generator",
  description: "Build grounded RFP responses from your company library.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <div className="min-h-screen flex flex-col">
            <TopNav />
            <main className="flex-1">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}

function TopNav() {
  return (
    <header className="bg-white/95 backdrop-blur border-b border-line sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center gap-7">
        <Link
          href="/knowledge"
          className="font-semibold text-ink text-lg tracking-tight flex items-center gap-2"
        >
          <span className="h-7 w-7 rounded bg-ink text-white text-xs grid place-items-center">
            RS
          </span>
          RFP Studio
        </Link>
        <nav className="flex gap-1 text-sm">
          <Link
            href="/knowledge"
            className="text-slate-600 hover:text-ink hover:bg-slate-100 font-medium px-3 py-2 rounded"
          >
            Library
          </Link>
          <Link
            href="/rfps"
            className="text-slate-600 hover:text-ink hover:bg-slate-100 font-medium px-3 py-2 rounded"
          >
            RFPs
          </Link>
        </nav>
      </div>
    </header>
  );
}
