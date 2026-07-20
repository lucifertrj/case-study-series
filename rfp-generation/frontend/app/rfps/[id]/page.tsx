"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  exportProject,
  generateAll,
  getProject,
  type ProjectDetail,
  type Question,
} from "../../lib/api";
import QuestionList from "../../components/QuestionList";
import AnswerEditor from "../../components/AnswerEditor";
import { useToast } from "../../components/Toast";

export default function RFPDetailPage() {
  const params = useParams<{ id: string }>();
  const projectId = params.id;
  const qc = useQueryClient();
  const toast = useToast();

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [generatingAll, setGeneratingAll] = useState(false);

  const { data: project, isLoading } = useQuery<ProjectDetail>({
    queryKey: ["project", projectId],
    queryFn: () => getProject(projectId),
    refetchInterval: (query) => {
      const p = query.state.data as ProjectDetail | undefined;
      if (!p) return false;
      if (p.status === "extracting" || p.status === "uploading") return 3000;
      if (generatingAll) return 3000;
      return false;
    },
  });

  useEffect(() => {
    if (!project) return;
    if (!selectedId && project.questions.length) {
      setSelectedId(project.questions[0].id);
    }
  }, [project, selectedId]);

  useEffect(() => {
    if (!project || !generatingAll) return;
    if (project.unanswered_count === 0) {
      setGeneratingAll(false);
      toast.success("All questions answered.");
    }
  }, [project, generatingAll, toast]);

  const generateAllMutation = useMutation({
    mutationFn: () => generateAll(projectId),
    onSuccess: () => {
      setGeneratingAll(true);
      toast.info("Generating answers in the background…");
    },
    onError: (err: unknown) => {
      toast.error(err instanceof Error ? err.message : String(err));
    },
  });

  const exportMutation = useMutation({
    mutationFn: () =>
      exportProject(
        projectId,
        project ? `${project.name.replace(/\s+/g, "_")}_response.docx` : "response.docx"
      ),
    onSuccess: () => toast.success("DOCX downloaded."),
    onError: (err: unknown) => {
      toast.error(err instanceof Error ? err.message : String(err));
    },
  });

  const selected: Question | null = useMemo(() => {
    if (!project || !selectedId) return null;
    return project.questions.find((q) => q.id === selectedId) || null;
  }, [project, selectedId]);

  if (isLoading || !project) {
    return <div className="p-6 text-sm text-slate-500">Loading project...</div>;
  }

  const reviewPct = project.total_questions
    ? Math.round((project.approved_count / project.total_questions) * 100)
    : 0;

  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col">
      <header className="border-b border-line bg-white px-6 py-4 flex items-center gap-5">
        <div className="min-w-0 flex-1">
          <h1 className="text-xl font-semibold text-ink truncate">
            {project.name}
          </h1>
          <p className="text-xs text-slate-500 truncate mt-1">
            Client: {project.client}
            {project.due_date &&
              ` · Due ${new Date(project.due_date).toLocaleDateString()}`}
          </p>
        </div>
        <div className="hidden md:block w-56">
          <div className="flex justify-between text-xs text-slate-500 mb-1">
            <span>Approved</span>
            <span>{reviewPct}%</span>
          </div>
          <div className="h-2 bg-slate-100 rounded overflow-hidden">
            <div className="h-full bg-emerald-500" style={{ width: `${reviewPct}%` }} />
          </div>
        </div>
        <button
          onClick={() => exportMutation.mutate()}
          disabled={exportMutation.isPending || project.approved_count === 0}
          className="text-sm px-4 py-2 rounded border border-line text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          title={project.approved_count === 0 ? "Approve at least one answer to enable export" : ""}
        >
          {exportMutation.isPending ? "Exporting..." : "Export DOCX"}
        </button>
      </header>

      {project.status === "extracting" && (
        <div className="bg-amber-50 border-b border-amber-200 text-amber-800 text-sm px-6 py-2">
          Extracting questions from the RFP file. This page will refresh automatically.
        </div>
      )}
      {project.status === "failed" && (
        <div className="bg-rose-50 border-b border-rose-200 text-rose-800 text-sm px-6 py-2">
          Failed to process RFP: {project.error_msg || "unknown error"}
        </div>
      )}

      <div className="flex-1 flex overflow-hidden">
        <aside className="w-80 flex-shrink-0 bg-white border-r border-line flex flex-col">
          <div className="p-3 border-b border-line">
            <button
              onClick={() => generateAllMutation.mutate()}
              disabled={
                generateAllMutation.isPending ||
                generatingAll ||
                project.unanswered_count === 0
              }
              className="w-full text-sm px-3 py-2 rounded bg-ink text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {generatingAll
                ? "Answering all..."
                : project.unanswered_count === 0
                ? "Nothing to answer"
                : `Answer all (${project.unanswered_count})`}
            </button>
          </div>
          <QuestionList
            questions={project.questions}
            selectedId={selectedId}
            onSelect={(q) => setSelectedId(q.id)}
          />
        </aside>

        <section className="flex-1 bg-slate-100/70 flex flex-col overflow-hidden">
          {selected ? (
            <AnswerEditor key={selected.id} question={selected} projectId={projectId} />
          ) : (
            <div className="p-8 text-sm text-slate-500">
              Select a question to view its answer.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
