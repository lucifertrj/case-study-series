"use client";

import { useState, FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createProject } from "../lib/api";
import { useToast } from "./Toast";

export default function NewRFPModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const toast = useToast();
  const [name, setName] = useState("");
  const [client, setClient] = useState("");
  const [dueDate, setDueDate] = useState("");
  const [file, setFile] = useState<File | null>(null);

  const create = useMutation({
    mutationFn: createProject,
    onSuccess: () => {
      toast.success("RFP created. Extracting questions...");
      qc.invalidateQueries({ queryKey: ["projects"] });
      setName("");
      setClient("");
      setDueDate("");
      setFile(null);
      onClose();
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err);
      toast.error(msg);
    },
  });

  if (!open) return null;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!file) {
      toast.error("Please upload the RFP file.");
      return;
    }
    create.mutate({
      name,
      client,
      due_date: dueDate || undefined,
      file,
    });
  };

  return (
    <div
      className="fixed inset-0 bg-slate-950/45 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded shadow-soft w-full max-w-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-5 border-b border-line">
          <div>
            <h2 className="text-lg font-semibold text-ink">New RFP</h2>
            <p className="text-sm text-slate-500 mt-1">
              Add the client document and project details.
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-800 text-xl leading-none"
          >
            ×
          </button>
        </div>

        <form onSubmit={submit} className="space-y-4 p-6">
          <div>
            <label className="text-sm font-medium text-slate-700">
              Project name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="mt-1 w-full border border-line rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
            />
          </div>

          <div>
            <label className="text-sm font-medium text-slate-700">Client</label>
            <input
              type="text"
              value={client}
              onChange={(e) => setClient(e.target.value)}
              required
              className="mt-1 w-full border border-line rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
            />
          </div>

          <div>
            <label className="text-sm font-medium text-slate-700">
              Due date
            </label>
            <input
              type="date"
              value={dueDate}
              onChange={(e) => setDueDate(e.target.value)}
              className="mt-1 w-full border border-line rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
            />
          </div>

          <div>
            <label className="text-sm font-medium text-slate-700">
              RFP document (PDF, DOCX, TXT, MD)
            </label>
            <input
              type="file"
              accept=".pdf,.docx,.txt,.md"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              required
              className="mt-1 block w-full rounded border border-line px-3 py-2 text-sm text-slate-700 file:mr-3 file:border-0 file:bg-slate-100 file:px-3 file:py-1 file:text-sm file:text-slate-700"
            />
          </div>

          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm rounded border border-line text-slate-700 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={create.isPending}
              className="px-4 py-2 text-sm rounded bg-ink text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {create.isPending ? "Creating..." : "Create RFP"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
