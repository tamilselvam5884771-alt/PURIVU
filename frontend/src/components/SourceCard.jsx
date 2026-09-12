import React from 'react';
import { FileText, Bookmark, Layers, Hash } from 'lucide-react';

export default function SourceCard({ source }) {
  if (!source) return null;

  const {
    document,
    document_title,
    page,
    category,
    is_number,
    clause_number,
    section_number,
    scheme_number,
    qco_reference,
    relevance_score
  } = source;

  return (
    <div className="bg-white rounded-xl p-4 flex flex-col justify-between space-y-3 border border-[#EEDFD7] text-left transition-all duration-200 hover:border-[#D98268]/40 hover:shadow-sm">
      {/* Header */}
      <div>
        <div className="flex items-start justify-between gap-2 mb-1.5">
          <div className="flex items-center space-x-2 min-w-0">
            <FileText className="w-4 h-4 text-[#4B1248] shrink-0" />
            <h4 className="font-semibold text-sm text-[#29242A] truncate" title={document_title || document}>
              {document_title || document}
            </h4>
          </div>
          <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#FFF9F5] text-[#4B1248] border border-[#EEDFD7] font-medium shrink-0">
            Pg {page}
          </span>
        </div>
        <p className="text-xs text-[#756873] font-mono truncate">{document}</p>
      </div>

      {/* Metadata Badges */}
      <div className="flex flex-wrap gap-1.5 text-[11px]">
        {category && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-[#FBE3D5]/50 text-[#4B1248] border border-[#EEDFD7]">
            <Layers className="w-3 h-3 text-[#D98268]" />
            {category}
          </span>
        )}
        {is_number && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-[#FFF9F5] text-[#4B1248] border border-[#EEDFD7] font-mono font-medium">
            <Bookmark className="w-3 h-3 text-[#2864E8]" />
            {is_number}
          </span>
        )}
        {clause_number && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-[#FFF9F5] text-[#756873] border border-[#EEDFD7] font-mono">
            <Hash className="w-3 h-3 text-[#756873]" />
            {clause_number}
          </span>
        )}
        {scheme_number && (
          <span className="inline-flex items-center px-2 py-0.5 rounded bg-[#FFF9F5] text-[#756873] border border-[#EEDFD7]">
            {scheme_number}
          </span>
        )}
        {qco_reference && (
          <span className="inline-flex items-center px-2 py-0.5 rounded bg-[#FBE3D5] text-[#4B1248] border border-[#EEDFD7] font-medium">
            QCO Ref
          </span>
        )}
      </div>

      {/* Relevance Meter */}
      {relevance_score !== undefined && relevance_score !== null && (
        <div className="pt-2 border-t border-[#EEDFD7] flex items-center justify-between text-[11px] text-[#756873]">
          <span>Match Score</span>
          <span className="font-mono font-medium text-[#4B1248]">L2: {relevance_score}</span>
        </div>
      )}
    </div>
  );
}
