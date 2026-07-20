"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { deleteDocument, getDocuments, type RfpDocument } from "../lib/api";
import { useToast } from "./Toast";

const STATUS_STYLES: Record<RfpDocument["status"], string> = {
  uploading: "bg-sky-100 text-sky-800",
  ingesting: "bg-amber-100 text-amber-800",
  indexed: "bg-emerald-100 text-emerald-800",
  failed: "bg-rose-100 text-rose-800",
};

const STATUS_LABEL: Record<RfpDocument["status"], string> = {
  uploading: "Uploading",
  ingesting: "Processing",
  indexed: "Ready",
  failed: "Failed",
};

export default function DocumentList() {
  const qc = useQueryClient();
  const toast = useToast();

  const { data: documents = [], isLoading } = useQuery({
    queryKey: ["documents"],
    queryFn: getDocuments,
    refetchInterval: (query) => {
      const docs = (query.state.data as RfpDocument[] | undefined) || [];
      const active = docs.some(
        (d) => d.status === "ingesting" || d.status === "uploading"
      );
      return active ? 3000 : false;
    },
  });

  const del = useMutation({
    mutationFn: (id: string) => deleteDocument(id),
    onSuccess: () => {
      toast.success("Document removed");
      qc.invalidateQueries({ queryKey: ["documents"] });
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err);
      toast.error(msg);
    },
  });

  if (isLoading) {
    return <div className="text-slate-500 text-sm">Loading documents...</div>;
  }

  if (!documents.length) {
    return (
      <div className="text-slate-600 text-sm bg-white border border-line rounded p-8 text-center shadow-sm">
        <div className="font-semibold text-ink">No documents yet</div>
        <div className="mt-1">Drop files above to build your company library.</div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-line rounded overflow-hidden shadow-sm">
      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-left text-slate-600">
          <tr>
            <th className="px-4 py-3 font-medium">Filename</th>
            <th className="px-4 py-2 font-medium">Type</th>
            <th className="px-4 py-2 font-medium">Status</th>
            <th className="px-4 py-2 font-medium text-right">Sections</th>
            <th className="px-4 py-2 font-medium">Uploaded</th>
            <th className="px-4 py-2"></th>
          </tr>
        </thead>
        <tbody>
          {documents.map((d) => (
            <tr key={d.id} className="border-t border-slate-100 hover:bg-slate-50/70">
              <td className="px-4 py-3 font-medium text-slate-800 truncate max-w-xs">
                {d.filename}
                {d.error_msg && (
                  <div
                    className="text-xs text-rose-700 truncate"
                    title={d.error_msg}
                  >
                    {d.error_msg}
                  </div>
                )}
              </td>
              <td className="px-4 py-2 uppercase text-slate-500 text-xs">
                {d.file_type}
              </td>
              <td className="px-4 py-2">
                <span
                  className={`px-2 py-0.5 rounded text-xs font-medium ${
                    STATUS_STYLES[d.status]
                  }`}
                >
                  {STATUS_LABEL[d.status]}
                </span>
              </td>
              <td className="px-4 py-2 text-right tabular-nums">
                {d.chunk_count}
              </td>
              <td className="px-4 py-2 text-slate-500">
                {new Date(d.created_at).toLocaleString()}
              </td>
              <td className="px-4 py-2 text-right">
                <button
                  onClick={() => del.mutate(d.id)}
                  disabled={del.isPending}
                  className="text-rose-600 hover:text-rose-800 text-xs"
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
