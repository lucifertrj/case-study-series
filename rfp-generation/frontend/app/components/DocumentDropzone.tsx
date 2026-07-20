"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { uploadDocument } from "../lib/api";
import { useToast } from "./Toast";

interface UploadItem {
  id: string;
  name: string;
  status: "pending" | "uploading" | "done" | "error";
  error?: string;
}

export default function DocumentDropzone() {
  const qc = useQueryClient();
  const toast = useToast();
  const [items, setItems] = useState<UploadItem[]>([]);

  const mutation = useMutation({
    mutationFn: (file: File) => uploadDocument(file),
  });

  const onDrop = useCallback(
    async (accepted: File[]) => {
      if (!accepted.length) return;

      const localItems: UploadItem[] = accepted.map((f) => ({
        id: `${f.name}-${Date.now()}-${Math.random()}`,
        name: f.name,
        status: "pending",
      }));
      setItems((prev) => [...localItems, ...prev]);

      for (const [i, file] of accepted.entries()) {
        const local = localItems[i];
        setItems((prev) =>
          prev.map((it) =>
            it.id === local.id ? { ...it, status: "uploading" } : it
          )
        );
        try {
          await mutation.mutateAsync(file);
          setItems((prev) =>
            prev.map((it) =>
              it.id === local.id ? { ...it, status: "done" } : it
            )
          );
          toast.success(`Uploaded ${file.name}`);
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : String(err);
          setItems((prev) =>
            prev.map((it) =>
              it.id === local.id
                ? { ...it, status: "error", error: msg }
                : it
            )
          );
          toast.error(`Failed to upload ${file.name}: ${msg}`);
        }
      }
      qc.invalidateQueries({ queryKey: ["documents"] });
    },
    [mutation, qc, toast]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        [".docx"],
      "text/plain": [".txt", ".md"],
    },
    multiple: true,
  });

  return (
    <div>
      <div
        {...getRootProps()}
        className={`border border-dashed rounded p-7 text-center cursor-pointer transition ${
          isDragActive
            ? "border-brand-500 bg-brand-50"
            : "border-line bg-white hover:border-slate-400"
        }`}
      >
        <input {...getInputProps()} />
        <p className="text-ink font-semibold">
          {isDragActive
            ? "Drop the files here..."
            : "Drop company documents here"}
        </p>
        <p className="text-sm text-slate-500 mt-1">
          PDF, DOCX, TXT, and Markdown are supported.
        </p>
      </div>

      {items.length > 0 && (
        <ul className="mt-4 space-y-1 text-sm">
          {items.map((it) => (
            <li
              key={it.id}
              className="flex items-center justify-between bg-white border border-line rounded px-3 py-2 shadow-sm"
            >
              <span className="truncate max-w-[70%]">{it.name}</span>
              <StatusPill status={it.status} error={it.error} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function StatusPill({
  status,
  error,
}: {
  status: UploadItem["status"];
  error?: string;
}) {
  const styles: Record<UploadItem["status"], string> = {
    pending: "bg-slate-100 text-slate-700",
    uploading: "bg-sky-100 text-sky-800 animate-pulse",
    done: "bg-emerald-100 text-emerald-800",
    error: "bg-rose-100 text-rose-800",
  };
  const labels: Record<UploadItem["status"], string> = {
    pending: "Queued",
    uploading: "Uploading...",
    done: "Uploaded",
    error: error || "Error",
  };
  return (
    <span
      title={error}
      className={`text-xs font-medium rounded px-2 py-0.5 ${styles[status]}`}
    >
      {labels[status]}
    </span>
  );
}
