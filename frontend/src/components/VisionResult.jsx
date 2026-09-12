import React, { useState } from 'react';
import { Eye, Edit3, Check, RefreshCw, ArrowRight, ShieldCheck, Sparkles } from 'lucide-react';
import AnswerCard from './AnswerCard';

export default function VisionResult({ visionData, imagePreviewUrl, onRequeryProduct, onReset }) {
  if (!visionData) return null;

  const { product, answer, evidence_status, sources } = visionData;

  const [isEditing, setIsEditing] = useState(false);
  const [correctedName, setCorrectedName] = useState(product?.name || '');
  const [correctedCategory, setCorrectedCategory] = useState(product?.category || '');

  const handleApplyCorrection = () => {
    if (!correctedName.trim()) return;
    setIsEditing(false);
    onRequeryProduct(correctedName.trim(), correctedCategory.trim());
  };

  const getConfidenceBadgeColor = (conf) => {
    switch (conf) {
      case 'High':
        return 'bg-emerald-50 text-emerald-800 border-emerald-200';
      case 'Moderate':
        return 'bg-blue-50 text-blue-800 border-blue-200';
      case 'Low':
      default:
        return 'bg-amber-50 text-amber-800 border-amber-200';
    }
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto text-left">
      {/* 1. PRODUCT UNDERSTANDING CARD */}
      <div className="bg-white rounded-2xl p-6 border border-[#EEDFD7] shadow-sm space-y-4">
        <div className="flex items-center justify-between border-b border-[#EEDFD7] pb-3">
          <div className="flex items-center space-x-2">
            <Eye className="w-5 h-5 text-[#4B1248]" />
            <h3 className="text-base sm:text-lg font-semibold text-[#29242A]">
              PRODUCT UNDERSTANDING
            </h3>
          </div>
          <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#FFF9F5] text-[#756873] border border-[#EEDFD7] font-mono">
            Vision Analysis
          </span>
        </div>

        <div className="flex flex-col sm:flex-row gap-5 items-start">
          {/* Image Preview Thumbnail */}
          {imagePreviewUrl && (
            <div className="w-28 h-28 sm:w-32 sm:h-32 rounded-xl overflow-hidden bg-[#FFF9F5] border border-[#EEDFD7] shrink-0 shadow-xs">
              <img
                src={imagePreviewUrl}
                alt={product?.name}
                className="w-full h-full object-cover"
              />
            </div>
          )}

          {/* Product Identification Details */}
          <div className="flex-1 space-y-3 min-w-0">
            {!isEditing ? (
              <>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <span className="text-xs uppercase tracking-wider text-[#D98268] font-semibold">
                      Likely Product
                    </span>
                    <h2 className="text-xl sm:text-2xl font-bold text-[#29242A] mt-0.5">
                      {product?.name}
                    </h2>
                  </div>
                  <span
                    className={`text-xs px-2.5 py-1 rounded-lg border font-semibold ${getConfidenceBadgeColor(
                      product?.confidence
                    )}`}
                  >
                    Vision Confidence: {product?.confidence}
                  </span>
                </div>

                <div className="text-xs text-[#756873] space-y-1">
                  <p>
                    <span className="font-semibold text-[#29242A]">Category:</span> {product?.category}
                  </p>
                  <p className="line-clamp-2">
                    <span className="font-semibold text-[#29242A]">Observation:</span> {product?.description}
                  </p>
                </div>

                {/* Product Confirmation / Edit Action */}
                <div className="pt-2 flex items-center space-x-3 text-xs">
                  <span className="text-[#756873] font-medium">Is this product identification correct?</span>
                  <button
                    onClick={() => setIsEditing(true)}
                    className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-[#FFF9F5] hover:bg-[#FBE3D5] text-[#4B1248] border border-[#EEDFD7] transition-colors cursor-pointer"
                  >
                    <Edit3 className="w-3.5 h-3.5" />
                    <span>Correct product name</span>
                  </button>
                </div>
              </>
            ) : (
              /* Correction Form */
              <div className="space-y-3 p-4 rounded-xl bg-[#FFF9F5] border border-[#EEDFD7]">
                <div className="flex items-center space-x-2 text-xs font-semibold text-[#4B1248]">
                  <Sparkles className="w-4 h-4 text-[#D98268]" />
                  <span>Correct Product Identification</span>
                </div>
                <div className="space-y-2">
                  <input
                    type="text"
                    value={correctedName}
                    onChange={(e) => setCorrectedName(e.target.value)}
                    placeholder="Enter correct product name (e.g. Rice Cooker)"
                    className="w-full bg-white rounded-lg px-3 py-2 text-xs text-[#29242A] outline-none border border-[#EEDFD7] focus:border-[#4B1248]"
                  />
                  <input
                    type="text"
                    value={correctedCategory}
                    onChange={(e) => setCorrectedCategory(e.target.value)}
                    placeholder="Enter category (e.g. Electrical Appliance)"
                    className="w-full bg-white rounded-lg px-3 py-2 text-xs text-[#29242A] outline-none border border-[#EEDFD7] focus:border-[#4B1248]"
                  />
                </div>
                <div className="flex items-center space-x-2 pt-1">
                  <button
                    onClick={handleApplyCorrection}
                    className="px-3 py-1.5 rounded-lg bg-[#4B1248] hover:bg-[#64175F] text-white text-xs font-medium transition-colors flex items-center space-x-1 cursor-pointer"
                  >
                    <Check className="w-3.5 h-3.5" />
                    <span>Search RAG with Correction</span>
                  </button>
                  <button
                    onClick={() => setIsEditing(false)}
                    className="px-3 py-1.5 rounded-lg bg-white text-[#756873] border border-[#EEDFD7] text-xs hover:text-[#29242A] transition-colors cursor-pointer"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 2. GROUNDED RAG ANSWER & SOURCES CARD */}
      <AnswerCard
        query={`BIS Requirements for ${product?.name}`}
        result={{ answer, evidence_status, sources }}
        onReset={onReset}
      />
    </div>
  );
}
