"use client";

import Link from "next/link";
import type { Project } from "../lib/api";

const STATUS_STYLES: Record<Project["status"], string> = {
  uploading: "bg-sky-100 text-sky-800",
  extracting: "bg-amber-100 text-amber-800 animate-pulse",
  ready: "bg-emerald-100 text-emerald-800",
  failed: "bg-rose-100 text-rose-800",
};

const STATUS_LABEL: Record<Project["status"], string> = {
  uploading: "Uploading",
  extracting: "Extracting questions…",
  ready: "Ready",
  failed: "Failed",
};

export default function ProjectCard({ project }: { project: Project }) {
  const total = project.total_questions || 0;
  const done = project.approved_count || 0;
  const pct = total ? Math.round((done / total) * 100) : 0;

  return (
    <Link
      href={`/rfps/${project.id}`}
      className="bg-white border border-line rounded p-5 hover:border-brand-500 hover:shadow-soft transition block"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h3 className="font-semibold text-slate-900 truncate">{project.name}</h3>
          <p className="text-sm text-slate-500 truncate">Client: {project.client}</p>
        </div>
        <span className={`text-xs font-medium px-2 py-1 rounded ${STATUS_STYLES[project.status]}`}>
          {STATUS_LABEL[project.status]}
        </span>
      </div>

      <div className="mt-4 flex items-center gap-3 text-xs text-slate-500">
        {project.due_date && (
          <span>Due {new Date(project.due_date).toLocaleDateString()}</span>
        )}
        <span className="truncate">{project.rfp_filename}</span>
      </div>

      <div className="mt-4">
        <div className="flex justify-between text-xs text-slate-500 mb-1">
          <span>{done} of {total} approved</span>
          <span>{pct}%</span>
        </div>
        <div className="h-2 bg-slate-100 rounded overflow-hidden">
          <div className="h-full bg-emerald-500 transition-all" style={{ width: `${pct}%` }} />
        </div>
      </div>

      {project.error_msg && (
        <div className="mt-3 text-xs text-rose-700 truncate" title={project.error_msg}>
          {project.error_msg}
        </div>
      )}
    </Link>
  );
}
