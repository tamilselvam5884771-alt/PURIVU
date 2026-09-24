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
          bg: 'bg-[#EAF7F1] text-[#287A5C] border-[#A9D9C5]',
          icon: ShieldCheck
        };
      case 'Moderate Confidence':
        return {
          label: 'Moderate Confidence',
          desc: 'Supported by retrieved BIS regulatory text',
          bg: 'bg-[#FFF3D9] text-[#9A6418] border-[#E7C477]',
          icon: ShieldCheck
        };
      case 'Out of Domain':
        return {
          label: 'Out of Domain',
          desc: 'Out of BIS regulatory domain',
          bg: 'bg-[#FBE5E0] text-[#9A4B3F] border-[#E5B3A8]',
          icon: ShieldAlert
        };
      case 'Limited Evidence':
      default:
        return {
          label: 'Limited Evidence',
          desc: 'Corpus has limited or no specific match',
          bg: 'bg-[#FFF3D9] text-[#9A6418] border-[#E7C477]',
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
      <div className="space-y-4 text-[#321F2F] leading-relaxed text-sm sm:text-base">
        {lines.map((line, idx) => {
          if (line.startsWith('### ')) {
            return (
              <h3 key={idx} className="text-base sm:text-lg font-bold text-[#5A1855] pt-3 border-t border-[#E8C5B5] first:pt-0 first:border-0 flex items-center gap-2 font-heading">
                <span className="w-1.5 h-1.5 rounded-full bg-[#E58F75]"></span>
                {line.replace('### ', '')}
              </h3>
            );
          } else if (line.startsWith('• ') || line.startsWith('* ')) {
            return (
              <div key={idx} className="flex items-start space-x-2 pl-2 my-1">
                <span className="text-[#E58F75] font-bold">•</span>
                <span>{line.replace(/^[•*]\s*/, '')}</span>
              </div>
            );
          } else if (/^\d+\.\s/.test(line)) {
            return (
              <div key={idx} className="flex items-start space-x-2 pl-2 my-1">
                <span className="font-mono text-[#E58F75] font-semibold">{line.match(/^\d+\./)[0]}</span>
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
    <div className="bg-[#FFFDFC] border border-[#E2B6A3] rounded-[20px] shadow-[0_4px_18px_rgba(90,24,85,0.06)] p-6 sm:p-8 max-w-4xl mx-auto space-y-6 text-left animate-fade-in">
      {/* Header & Confidence Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[#E8C5B5]">
        <div>
          <span className="text-xs uppercase tracking-wider text-[#735F6C] font-semibold font-mono">
            Grounded Answer
          </span>
          <h2 className="text-xl sm:text-2xl font-bold text-[#5A1855] mt-1 font-heading">
            "{query}"
          </h2>
        </div>

        <div className={`inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-xl border ${badgeInfo.bg} shrink-0`}>
          <BadgeIcon className="w-4 h-4" />
          <span className="text-xs font-semibold">{badgeInfo.label}</span>
        </div>
      </div>

      {/* Answer Content */}
      <div className="p-6 rounded-2xl bg-[#FFF9F5] border border-[#E8C5B5]">
        {renderFormattedAnswer(answer)}
      </div>

      {/* Sources Grid */}
      {sources && sources.length > 0 && (
        <div className="space-y-3 pt-4 border-t border-[#E8C5B5]">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-[#5A1855] font-semibold text-sm">
              <BookOpen className="w-4 h-4 text-[#E58F75]" />
              <span>Evidence Sources</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-[#FBE0D2] border border-[#E7BDAA] text-[#6A2365] font-mono font-semibold">
                {sources.length}
              </span>
            </div>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#FFF9F5] border border-[#E2B6A3] text-[#735F6C] font-mono">
              Local BIS PDF Corpus
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-1">
            {sources.map((src, idx) => (
              <SourceCard key={idx} source={src} />
            ))}
          </div>
        </div>
      )}

      {/* Footer Controls */}
      <div className="flex items-center justify-between pt-4 border-t border-[#E8C5B5] text-xs">
        <div className="flex items-center space-x-2">
          <button
            onClick={handleCopy}
            className="inline-flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-[#FFF9F5] hover:bg-[#FBE0D2] text-[#5A1855] border border-[#E2B6A3] hover:border-[#D58A72] transition-all cursor-pointer font-medium"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-[#3A9B78]" /> : <Copy className="w-3.5 h-3.5 text-[#E58F75]" />}
            <span>{copied ? 'Copied' : 'Copy Answer'}</span>
          </button>

          {'speechSynthesis' in window && (
            <button
              onClick={handleSpeak}
              className={`inline-flex items-center space-x-1.5 px-3.5 py-2 rounded-xl border transition-all cursor-pointer font-medium ${
                speaking
                  ? 'bg-[#5A1855] text-white border-[#5A1855]'
                  : 'bg-[#FFF9F5] hover:bg-[#FBE0D2] text-[#5A1855] border-[#E2B6A3] hover:border-[#D58A72]'
              }`}
            >
              <span>{speaking ? '🔇 Stop Reading' : '🔊 Read Answer'}</span>
            </button>
          )}
        </div>

        <button
          onClick={onReset}
          className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-[#5A1855] hover:bg-[#6A2365] text-white font-medium shadow-xs hover:shadow-sm transition-all cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>New Query</span>
        </button>
      </div>
    </div>
  );
}
