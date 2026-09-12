import React from 'react';
import { Camera, Mic, Sparkles, X, ArrowRight } from 'lucide-react';

export default function PlaceholderModal({ type, onClose }) {
  if (!type) return null;

  const isVision = type === 'show';
  const title = isVision ? 'SHOW — Product Image Recognition' : 'SPEAK — Voice Assistant';
  const subtitle = isVision
    ? 'Upload or capture a product label to automatically identify Indian Standard (IS) requirements.'
    : 'Speak naturally in English or regional languages to query BIS regulations.';
  const stepText = isVision ? 'Scheduled for Step 6' : 'Scheduled for Step 7';
  const Icon = isVision ? Camera : Mic;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-fade-in">
      <div className="glass-panel max-w-md w-full rounded-2xl p-6 border border-indigo-500/40 shadow-2xl relative text-left">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-indigo-400 hover:text-white hover:bg-indigo-900/50 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center space-x-3 mb-4">
          <div className="p-3 rounded-xl bg-indigo-600/30 border border-indigo-500/40 text-indigo-300">
            <Icon className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-indigo-100">{title}</h3>
            <span className="text-xs px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-700/40 font-mono">
              {stepText}
            </span>
          </div>
        </div>

        <p className="text-sm text-indigo-200/80 mb-6 leading-relaxed">
          {subtitle}
        </p>

        <div className="p-4 rounded-xl bg-indigo-950/60 border border-indigo-800/40 mb-6 text-xs text-indigo-300 space-y-2">
          <div className="flex items-center space-x-2 font-semibold text-indigo-200">
            <Sparkles className="w-4 h-4 text-indigo-400 shrink-0" />
            <span>Planned Architecture</span>
          </div>
          <p className="text-indigo-300/70">
            {isVision
              ? 'Image → Vision Model → Product Features → PURIVU RAG Engine → BIS Citations'
              : 'Voice → Speech-to-Text → Language Detection → PURIVU RAG Engine → Answer'}
          </p>
        </div>

        <button
          onClick={onClose}
          className="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition-all flex items-center justify-center space-x-2 shadow-lg shadow-indigo-600/30"
        >
          <span>Continue with ASK (Text Search)</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
