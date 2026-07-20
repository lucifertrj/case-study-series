"use client";

import { useState } from "react";
import type { AnswerSource } from "../lib/api";

export default function SourceChip({ source }: { source: AnswerSource }) {
  const [open, setOpen] = useState(false);
  const label =
    source.document_name ||
    (source.locator ? `chunk ${source.locator.slice(0, 8)}` : "chunk");

  return (
    <div className="relative inline-block">
      <button
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 bg-slate-100 hover:bg-slate-200 border border-line rounded px-2.5 py-1 text-xs text-slate-700"
      >
        <span className="font-medium truncate max-w-[10rem]">{label}</span>
        {source.locator && (
          <span className="text-slate-400">·{source.locator.slice(0, 6)}</span>
        )}
        <span className="text-slate-400">
          {source.score ? source.score.toFixed(2) : ""}
        </span>
      </button>

      {open && (
        <div className="absolute z-30 mt-2 w-[28rem] max-w-[80vw] bg-white border border-line rounded shadow-soft p-4 text-xs text-slate-700">
          <div className="flex justify-between mb-2 items-center">
            <span className="font-semibold text-slate-800 truncate max-w-[70%]">
              {source.document_name || "Unknown document"}
            </span>
            <button
              onClick={() => setOpen(false)}
              className="text-slate-400 hover:text-slate-700"
            >
              ×
            </button>
          </div>
          <div className="max-h-64 overflow-y-auto whitespace-pre-wrap leading-relaxed text-slate-600">
            {source.chunk_text}
          </div>
        </div>
      )}
    </div>
  );
}
