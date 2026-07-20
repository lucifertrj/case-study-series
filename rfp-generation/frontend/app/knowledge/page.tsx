"use client";

import DocumentDropzone from "../components/DocumentDropzone";
import DocumentList from "../components/DocumentList";

export default function KnowledgePage() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-7">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-brand-600">
            Company library
          </p>
          <h1 className="mt-2 text-2xl font-semibold text-ink">
            Source material
          </h1>
          <p className="text-sm text-slate-600 mt-2 max-w-xl">
            Upload policy documents, past responses, case studies, and product
            notes. Answers will cite the documents they use.
          </p>
        </div>
      </div>

      <div className="mb-8">
        <DocumentDropzone />
      </div>

      <div>
        <h2 className="text-sm font-semibold text-slate-800 mb-3">
          Library documents
        </h2>
        <DocumentList />
      </div>
    </div>
  );
}
