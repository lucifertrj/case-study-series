"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getProjects, type Project } from "../lib/api";
import ProjectCard from "../components/ProjectCard";
import NewRFPModal from "../components/NewRFPModal";

export default function RFPsPage() {
  const [open, setOpen] = useState(false);

  const { data: projects = [], isLoading } = useQuery({
    queryKey: ["projects"],
    queryFn: getProjects,
    refetchInterval: (query) => {
      const projects = (query.state.data as Project[] | undefined) || [];
      const active = projects.some(
        (p) => p.status === "extracting" || p.status === "uploading"
      );
      return active ? 3000 : false;
    },
  });

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between mb-7">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-brand-600">
            Response workspace
          </p>
          <h1 className="mt-2 text-3xl font-semibold text-ink">RFPs</h1>
          <p className="text-sm text-slate-600 mt-2 max-w-2xl">
            Create a new response by uploading the client RFP. Questions are
            pulled out automatically so your team can draft, review, and export.
          </p>
        </div>
        <button
          onClick={() => setOpen(true)}
          className="bg-ink hover:bg-slate-800 text-white text-sm font-medium px-4 py-2.5 rounded shadow-sm"
        >
          New RFP
        </button>
      </div>

      {isLoading ? (
        <div className="text-slate-500 text-sm">Loading projects...</div>
      ) : projects.length === 0 ? (
        <div className="bg-white border border-line rounded p-10 text-center text-slate-600 text-sm shadow-sm">
          <div className="mx-auto h-12 w-12 rounded bg-slate-100 grid place-items-center text-slate-500 mb-3">
            RFP
          </div>
          <div className="font-semibold text-ink">No RFPs yet</div>
          <div className="mt-1">
            Create one to extract questions and start drafting responses.
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {projects.map((p) => (
            <ProjectCard key={p.id} project={p} />
          ))}
        </div>
      )}

      <NewRFPModal open={open} onClose={() => setOpen(false)} />
    </div>
  );
}
