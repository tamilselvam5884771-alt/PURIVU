import React, { useState } from 'react';
import { ShieldCheck, ShieldAlert, BookOpen, Check, Copy, RefreshCw } from 'lucide-react';
import SourceCard from './SourceCard';

export default function AnswerCard({ query, result, onReset }) {
  const [copied, setCopied] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  if (!result) return null;

  const { answer, evidence_status, sources } = result;

  const getConfidenceBadge = (status) => {
    switch (status) {
      case 'High Confidence':
        return {
          label: 'High Confidence',
          desc: 'Verified by top-ranked BIS document evidence',
          bg: 'bg-emerald-50 text-emerald-800 border-emerald-200',
          icon: ShieldCheck
        };
      case 'Moderate Confidence':
        return {
          label: 'Moderate Confidence',
          desc: 'Supported by retrieved BIS regulatory text',
          bg: 'bg-[#FBE3D5] text-[#4B1248] border-[#EEDFD7]',
          icon: ShieldCheck
        };
      case 'Limited Evidence':
      default:
        return {
          label: 'Limited Evidence',
          desc: 'Corpus has limited or no specific match',
          bg: 'bg-amber-50 text-amber-800 border-amber-200',
          icon: ShieldAlert
        };
    }
  };

  const badgeInfo = getConfidenceBadge(evidence_status);
  const BadgeIcon = badgeInfo.icon;

  const handleCopy = () => {
    navigator.clipboard.writeText(answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSpeak = () => {
    if (!('speechSynthesis' in window)) return;
    if (speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
      return;
    }

    const cleanText = answer.replace(/[#*•\d\.]/g, '').trim();
    const utterance = new SpeechSynthesisUtterance(cleanText);

    let langTag = 'en-IN';
    const isTamil = result.response_language === 'Tamil' || Array.from(cleanText).some(c => c >= '\u0b80' && c <= '\u0bff');
    const isHindi = result.response_language === 'Hindi' || Array.from(cleanText).some(c => c >= '\u0900' && c <= '\u097f');
    if (isTamil) {
      langTag = 'ta-IN';
    } else if (isHindi) {
      langTag = 'hi-IN';
    }
    utterance.lang = langTag;

    const voices = window.speechSynthesis.getVoices();
    const matchingVoice = voices.find(v => v.lang.includes(langTag) || v.lang.replace('_', '-').startsWith(langTag.split('-')[0]));
    if (matchingVoice) {
      utterance.voice = matchingVoice;
    }

    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);

    setSpeaking(true);
    window.speechSynthesis.speak(utterance);
  };

  const renderFormattedAnswer = (text) => {
    if (!text) return null;
    
    const lines = text.split('\n');
    return (
      <div className="space-y-4 text-[#29242A] leading-relaxed text-sm sm:text-base">
        {lines.map((line, idx) => {
          if (line.startsWith('### ')) {
            return (
              <h3 key={idx} className="text-base sm:text-lg font-bold text-[#4B1248] pt-3 border-t border-[#EEDFD7] first:pt-0 first:border-0 flex items-center gap-2 font-heading">
                <span className="w-1.5 h-1.5 rounded-full bg-[#D98268]"></span>
                {line.replace('### ', '')}
              </h3>
            );
          } else if (line.startsWith('• ') || line.startsWith('* ')) {
            return (
              <div key={idx} className="flex items-start space-x-2 pl-2 my-1">
                <span className="text-[#D98268] font-bold">•</span>
                <span>{line.replace(/^[•*]\s*/, '')}</span>
              </div>
            );
          } else if (/^\d+\.\s/.test(line)) {
            return (
              <div key={idx} className="flex items-start space-x-2 pl-2 my-1">
                <span className="font-mono text-[#D98268] font-semibold">{line.match(/^\d+\./)[0]}</span>
                <span>{line.replace(/^\d+\.\s*/, '')}</span>
              </div>
            );
          } else if (line.trim()) {
            return <p key={idx} className="my-1.5">{line}</p>;
          }
          return null;
        })}
      </div>
    );
  };

  return (
    <div className="purivu-card p-6 sm:p-8 max-w-4xl mx-auto space-y-6 text-left animate-fade-in">
      {/* Header & Confidence Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[#EEDFD7]">
        <div>
          <span className="text-xs uppercase tracking-wider text-[#756873] font-semibold font-mono">
            Grounded Answer
          </span>
          <h2 className="text-xl sm:text-2xl font-bold text-[#4B1248] mt-1 font-heading">
            "{query}"
          </h2>
        </div>

        <div className={`inline-flex items-center space-x-2 px-3 py-1.5 rounded-xl border ${badgeInfo.bg} shrink-0`}>
          <BadgeIcon className="w-4 h-4" />
          <span className="text-xs font-semibold">{badgeInfo.label}</span>
        </div>
      </div>

      {/* Answer Content */}
      <div className="p-6 rounded-2xl bg-[#FFF9F5] border border-[#EEDFD7]">
        {renderFormattedAnswer(answer)}
      </div>

      {/* Sources Grid */}
      {sources && sources.length > 0 && (
        <div className="space-y-3 pt-4 border-t border-[#EEDFD7]">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-[#4B1248] font-semibold text-sm">
              <BookOpen className="w-4 h-4 text-[#D98268]" />
              <span>Evidence Sources ({sources.length})</span>
            </div>
            <span className="text-xs text-[#756873] font-mono">
              Local BIS PDF Corpus
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            {sources.map((src, idx) => (
              <SourceCard key={idx} source={src} />
            ))}
          </div>
        </div>
      )}

      {/* Footer Controls */}
      <div className="flex items-center justify-between pt-4 border-t border-[#EEDFD7] text-xs">
        <div className="flex items-center space-x-2">
          <button
            onClick={handleCopy}
            className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-[#FFF9F5] hover:bg-[#FBE3D5] text-[#4B1248] border border-[#EEDFD7] transition-colors cursor-pointer"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5 text-[#D98268]" />}
            <span>{copied ? 'Copied' : 'Copy Answer'}</span>
          </button>

          {'speechSynthesis' in window && (
            <button
              onClick={handleSpeak}
              className={`inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl border transition-colors cursor-pointer ${
                speaking
                  ? 'bg-[#4B1248] text-white border-[#4B1248]'
                  : 'bg-[#FFF9F5] hover:bg-[#FBE3D5] text-[#4B1248] border-[#EEDFD7]'
              }`}
            >
              <span>{speaking ? '🔇 Stop Reading' : '🔊 Read Answer'}</span>
            </button>
          )}
        </div>

        <button
          onClick={onReset}
          className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-[#4B1248] hover:bg-[#64175F] text-white font-medium shadow-sm transition-all cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>New Query</span>
        </button>
      </div>
    </div>
  );
}
