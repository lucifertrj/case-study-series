// Home redirects to /knowledge via next.config.ts.
// This file exists so that /app has an index route in dev previews.
export default function HomePage() {
  return (
    <div className="p-8 text-slate-600">
      Redirecting to the company library...
    </div>
  );
}
