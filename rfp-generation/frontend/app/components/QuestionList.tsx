"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";
import type { Question, QuestionStatus } from "../lib/api";

type Filter = "all" | QuestionStatus;

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "unanswered", label: "Unanswered" },
  { key: "draft", label: "Draft" },
  { key: "approved", label: "Approved" },
];

const STATUS_ICON: Record<QuestionStatus, string> = {
  unanswered: "",
  draft: "",
  approved: "",
};

const STATUS_COLOR: Record<QuestionStatus, string> = {
  unanswered: "bg-slate-300",
  draft: "bg-amber-500",
  approved: "bg-emerald-500",
};

export default function QuestionList({
  questions,
  selectedId,
  onSelect,
}: {
  questions: Question[];
  selectedId: string | null;
  onSelect: (q: Question) => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");

  const filtered = useMemo(() => {
    if (filter === "all") return questions;
    return questions.filter((q) => q.status === filter);
  }, [questions, filter]);

  return (
    <div className="flex flex-col h-full">
      <div className="grid grid-cols-2 gap-1 p-2 border-b border-line text-xs">
        {FILTERS.map((f) => {
          const count =
            f.key === "all"
              ? questions.length
              : questions.filter((q) => q.status === f.key).length;
          return (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={clsx(
                "py-2 font-medium transition rounded",
                filter === f.key
                  ? "bg-slate-900 text-white"
                  : "text-slate-500 hover:text-slate-700 hover:bg-slate-100"
              )}
            >
              {f.label} ({count})
            </button>
          );
        })}
      </div>

      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="p-4 text-sm text-slate-500">
            No questions in this view.
          </div>
        ) : (
          <ul>
            {filtered.map((q) => (
              <li key={q.id}>
                <button
                  onClick={() => onSelect(q)}
                  className={clsx(
                    "w-full text-left px-3 py-3 border-b border-slate-100 flex gap-3 items-start hover:bg-slate-50 transition",
                    selectedId === q.id && "bg-brand-50"
                  )}
                >
                  <span
                    className={clsx(
                      "mt-1 h-2.5 w-2.5 rounded-full flex-shrink-0",
                      STATUS_COLOR[q.status]
                    )}
                    title={q.status}
                  >
                    {STATUS_ICON[q.status]}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium text-slate-500">
                      Q{q.number}
                    </div>
                    <div className="text-sm text-slate-800 line-clamp-3 leading-snug">
                      {q.text}
                    </div>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
