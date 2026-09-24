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
    <div className="bg-[#FFFDFC] rounded-[15px] p-4 flex flex-col justify-between space-y-3 border border-[#E2B6A3] text-left transition-all duration-200 hover:border-[#D58A72] hover:shadow-[0_5px_16px_rgba(90,24,85,0.08)] hover:-translate-y-0.5">
      {/* Header */}
      <div>
        <div className="flex items-start justify-between gap-2 mb-1.5">
          <div className="flex items-center space-x-2 min-w-0">
            <FileText className="w-4 h-4 text-[#5A1855] shrink-0" />
            <h4 className="font-semibold text-sm text-[#321F2F] truncate" title={document_title || document}>
              {document_title || document}
            </h4>
          </div>
          <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#FFF9F5] text-[#6A2365] border border-[#E2B6A3] font-medium font-mono shrink-0">
            Pg {page}
          </span>
        </div>
        <p className="text-xs text-[#735F6C] font-mono truncate">{document}</p>
      </div>

      {/* Metadata Badges */}
      <div className="flex flex-wrap gap-1.5 text-[11px]">
        {category && (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md bg-[#FBE0D2] text-[#6A2365] border border-[#E7BDAA] font-medium">
            <Layers className="w-3 h-3 text-[#E58F75]" />
            {category}
          </span>
        )}
        {is_number && (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md bg-[#FFF9F5] text-[#3478E5] border border-[#E2B6A3] font-mono font-medium">
            <Bookmark className="w-3 h-3 text-[#3478E5]" />
            {is_number}
          </span>
        )}
        {clause_number && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-[#FFF9F5] text-[#735F6C] border border-[#E2B6A3] font-mono">
            <Hash className="w-3 h-3 text-[#735F6C]" />
            {clause_number}
          </span>
        )}
        {scheme_number && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-[#FFF9F5] text-[#735F6C] border border-[#E2B6A3]">
            {scheme_number}
          </span>
        )}
        {qco_reference && (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-md bg-[#FBE0D2] text-[#6A2365] border border-[#E7BDAA] font-medium">
            QCO Ref
          </span>
        )}
      </div>

      {/* Relevance Meter */}
      {relevance_score !== undefined && relevance_score !== null && (
        <div className="pt-2 border-t border-[#E8C5B5] flex items-center justify-between text-[11px] text-[#735F6C]">
          <span>Match Score</span>
          <span className="font-mono font-semibold text-[#5A1855]">L2: {relevance_score}</span>
        </div>
      )}
    </div>
  );
}
