"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import {
  generateAnswer,
  updateAnswer,
  updateQuestionStatus,
  type Question,
} from "../lib/api";
import SourceChip from "./SourceChip";
import { useToast } from "./Toast";

export default function AnswerEditor({
  question,
  projectId,
}: {
  question: Question;
  projectId: string;
}) {
  const qc = useQueryClient();
  const toast = useToast();
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const answer = question.answer ?? null;
  const [content, setContent] = useState<string>(answer?.content ?? "");
  const [dirty, setDirty] = useState(false);

  // Sync textarea when question/answer changes (e.g. after generate or polling).
  useEffect(() => {
    setContent(question.answer?.content ?? "");
    setDirty(false);
  }, [question.id, question.answer?.content]);

  // Auto-resize textarea.
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.max(ta.scrollHeight, 160)}px`;
  }, [content]);

  const invalidate = () =>
    qc.invalidateQueries({ queryKey: ["project", projectId] });

  const genMutation = useMutation({
    mutationFn: () => generateAnswer(question.id),
    onSuccess: () => {
      toast.success("Answer generated.");
      invalidate();
    },
    onError: (err: unknown) => {
      toast.error(err instanceof Error ? err.message : String(err));
    },
  });

  const saveMutation = useMutation({
    mutationFn: (v: string) => updateAnswer(question.id, v),
    onSuccess: () => {
      setDirty(false);
      toast.info("Saved.");
      invalidate();
    },
    onError: (err: unknown) => {
      toast.error(err instanceof Error ? err.message : String(err));
    },
  });

  const statusMutation = useMutation({
    mutationFn: (status: Question["status"]) =>
      updateQuestionStatus(question.id, status),
    onSuccess: () => invalidate(),
    onError: (err: unknown) => {
      toast.error(err instanceof Error ? err.message : String(err));
    },
  });

  const onBlur = () => {
    if (dirty) saveMutation.mutate(content);
  };

  const approve = () =>
    statusMutation.mutate(question.status === "approved" ? "draft" : "approved");

  return (
    <div className="p-6 flex-1 flex flex-col overflow-y-auto">
      <div className="bg-white border border-line rounded shadow-sm">
        <div className="p-5 border-b border-line">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium text-slate-500">
              Question {question.number}
            </span>
            <StatusBadge status={question.status} />
          </div>
          <h2 className="text-xl font-semibold text-ink leading-snug">
            {question.text}
          </h2>
        </div>

        <div className="p-5">
          <div className="flex flex-wrap gap-2 mb-4">
            <button
              onClick={() => genMutation.mutate()}
              disabled={genMutation.isPending}
              className="px-4 py-2 rounded bg-ink text-white text-sm font-medium hover:bg-slate-800 disabled:opacity-50"
            >
              {genMutation.isPending ? "Generating..." : answer ? "Regenerate" : "Generate"}
            </button>
            <button
              onClick={approve}
              disabled={!answer || statusMutation.isPending}
              className={clsx(
                "px-4 py-2 rounded text-sm font-medium border",
                question.status === "approved"
                  ? "bg-white border-emerald-300 text-emerald-700 hover:bg-emerald-50"
                  : "bg-emerald-600 text-white border-emerald-600 hover:bg-emerald-700 disabled:opacity-50"
              )}
            >
              {question.status === "approved" ? "Move to draft" : "Approve"}
            </button>
          </div>

          <textarea
            ref={textareaRef}
            value={content}
            onChange={(e) => {
              setContent(e.target.value);
              setDirty(true);
            }}
            onBlur={onBlur}
            placeholder={answer ? "Edit the answer. Changes save on blur." : "No answer yet."}
            className="w-full min-h-[18rem] resize-none border border-line rounded p-4 text-sm leading-6 text-slate-800 bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-brand-500"
          />
          {dirty && (
            <div className="text-xs text-amber-600 mt-2">
              Unsaved changes will save when this field loses focus.
            </div>
          )}

          {answer && answer.sources.length > 0 && (
            <div className="mt-5">
              <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                Sources
              </div>
              <div className="flex flex-wrap gap-2">
                {answer.sources.map((s, i) => (
                  <SourceChip key={i} source={s} />
                ))}
              </div>
            </div>
          )}

          {answer && (
            <div className="text-xs text-slate-400 mt-5">
              Generated by {answer.generated_by} · updated{" "}
              {new Date(answer.updated_at).toLocaleString()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: Question["status"] }) {
  const styles: Record<Question["status"], string> = {
    unanswered: "bg-slate-100 text-slate-700",
    draft: "bg-amber-100 text-amber-800",
    approved: "bg-emerald-100 text-emerald-800",
  };
  return (
    <span className={clsx("text-xs px-2 py-1 rounded font-medium capitalize", styles[status])}>
      {status}
    </span>
  );
}
