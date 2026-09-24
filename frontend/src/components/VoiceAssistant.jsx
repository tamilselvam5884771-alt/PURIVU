import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Globe, Sparkles, Loader2, AlertCircle } from 'lucide-react';

export default function VoiceAssistant({ onVoiceQuery, loading }) {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [selectedLang, setSelectedLang] = useState('Auto');
  const [speechSupported, setSpeechSupported] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  const recognitionRef = useRef(null);

  const languages = [
    { code: 'Auto', label: 'Auto Detect', langTag: 'en-IN' },
    { code: 'English', label: 'English', langTag: 'en-IN' },
    { code: 'Tamil', label: 'தமிழ் (Tamil)', langTag: 'ta-IN' },
    { code: 'Hindi', label: 'हिन्दी (Hindi)', langTag: 'hi-IN' }
  ];

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechSupported(false);
      return;
    }

    const rec = new SpeechRecognition();
    rec.continuous = false;
    rec.interimResults = true;

    rec.onstart = () => {
      setIsListening(true);
      setErrorMsg('');
    };

    rec.onresult = (event) => {
      let currentTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        currentTranscript += event.results[i][0].transcript;
      }
      setTranscript(currentTranscript);
    };

    rec.onerror = (event) => {
      setIsListening(false);
      if (event.error === 'not-allowed') {
        setErrorMsg('Microphone access was denied. Please allow microphone permissions.');
      } else if (event.error === 'no-speech') {
        setErrorMsg('No speech detected. Please try again.');
      } else {
        setErrorMsg(`Speech error: ${event.error}`);
      }
    };

    rec.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = rec;

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, []);

  const toggleListening = () => {
    if (!speechSupported) return;

    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    } else {
      setTranscript('');
      setErrorMsg('');
      const targetLang = languages.find(l => l.code === selectedLang)?.langTag || 'en-IN';
      if (recognitionRef.current) {
        recognitionRef.current.lang = targetLang;
        try {
          recognitionRef.current.start();
        } catch (e) {
          recognitionRef.current.stop();
          setTimeout(() => recognitionRef.current?.start(), 200);
        }
      }
    }
  };

  const handleSendVoiceQuery = () => {
    if (!transcript.trim()) return;
    onVoiceQuery(transcript.trim(), selectedLang);
  };

  return (
    <div className="max-w-2xl mx-auto w-full bg-[#FFFDFC] rounded-[20px] p-6 sm:p-8 border border-[#E2B6A3] shadow-[0_4px_18px_rgba(90,24,85,0.06)] text-center animate-fade-in space-y-6">
      {/* Header & Language Selector */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 border-b border-[#E8C5B5] pb-4">
        <div className="text-left">
          <h2 className="text-xl sm:text-2xl font-bold text-[#5A1855] font-heading">
            Speak Naturally
          </h2>
          <p className="text-xs text-[#735F6C] font-mono mt-0.5">
            English • தமிழ் • हिन्दी
          </p>
        </div>

        {/* Language Selector Dropdown */}
        <div className="flex items-center space-x-2 bg-[#FFF9F5] px-3.5 py-1.5 rounded-xl border border-[#E2B6A3]">
          <Globe className="w-4 h-4 text-[#5A1855] shrink-0" />
          <select
            value={selectedLang}
            onChange={(e) => setSelectedLang(e.target.value)}
            disabled={isListening || loading}
            className="bg-transparent border-0 outline-none text-xs sm:text-sm text-[#321F2F] font-semibold cursor-pointer"
          >
            {languages.map((l) => (
              <option key={l.code} value={l.code} className="bg-[#FFFDFC] text-[#321F2F]">
                {l.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Microphone Surface */}
      {speechSupported ? (
        <div className="py-6 space-y-6">
          {/* Animated Central Microphone */}
          <div className="relative flex items-center justify-center">
            {isListening && (
              <>
                <div className="absolute w-36 h-36 rounded-full border border-[#5A1855]/40 animate-ring-1 pointer-events-none" />
                <div className="absolute w-48 h-48 rounded-full border border-[#E58F75]/30 animate-ring-2 pointer-events-none" />
                <div className="absolute w-28 h-28 rounded-full bg-[#5A1855]/10 blur-md animate-pulse pointer-events-none" />
              </>
            )}

            <button
              type="button"
              onClick={toggleListening}
              disabled={loading}
              className={`relative z-10 w-24 h-24 rounded-full flex flex-col items-center justify-center transition-all duration-300 cursor-pointer ${
                isListening
                  ? 'bg-rose-600 hover:bg-rose-700 text-white shadow-md scale-105 animate-pulse'
                  : 'bg-[#5A1855] hover:bg-[#6A2365] text-white shadow-xs hover:scale-105'
              }`}
            >
              {isListening ? (
                <>
                  <MicOff className="w-8 h-8" />
                  <span className="text-[10px] font-bold tracking-wider mt-1">STOP</span>
                </>
              ) : (
                <>
                  <Mic className="w-9 h-9" />
                  <span className="text-[10px] font-bold tracking-wider mt-1">SPEAK</span>
                </>
              )}
            </button>
          </div>

          {/* Status Indicator */}
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-widest text-[#5A1855] font-mono">
              {isListening ? '● LISTENING' : loading ? 'CHECKING BIS EVIDENCE' : 'Tap microphone & speak'}
            </p>
          </div>

          {/* Real-time Transcript */}
          {transcript && (
            <div className="p-4 rounded-xl bg-[#FFF9F5] border border-[#E8C5B5] text-left space-y-2 animate-fade-in">
              <div className="flex items-center justify-between text-xs font-mono text-[#735F6C]">
                <span>TRANSCRIPT:</span>
                <span>Language: {selectedLang}</span>
              </div>
              <p className="text-base text-[#321F2F] font-medium italic">
                "{transcript}"
              </p>
            </div>
          )}

          {/* Action Button */}
          {transcript && !isListening && (
            <button
              onClick={handleSendVoiceQuery}
              disabled={loading}
              className="w-full py-3 rounded-xl bg-[#5A1855] hover:bg-[#6A2365] text-white font-semibold text-sm transition-all flex items-center justify-center space-x-2 shadow-xs cursor-pointer"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin text-white" />
                  <span>CHECKING BIS EVIDENCE...</span>
                </>
              ) : (
                <>
                  <span>Understand Query</span>
                  <Sparkles className="w-4 h-4 text-[#E58F75]" />
                </>
              )}
            </button>
          )}
        </div>
      ) : (
        /* Fallback for Unsupported Browsers */
        <div className="p-6 rounded-xl bg-[#FFF3D9] border border-[#E7C477] text-[#9A6418] text-left space-y-2">
          <div className="flex items-center space-x-2 font-semibold text-sm">
            <AlertCircle className="w-5 h-5 text-[#9A6418]" />
            <span>Web Speech Recognition Unavailable</span>
          </div>
          <p className="text-xs text-[#9A6418]">
            Your browser does not support Web Speech Recognition. You can still use <strong>ASK</strong> mode to type your question.
          </p>
        </div>
      )}

      {/* Error Message */}
      {errorMsg && (
        <div className="p-3.5 rounded-xl bg-[#FBE5E0] border border-[#E5B3A8] text-[#9A4B3F] text-xs sm:text-sm text-left font-medium">
          ⚠️ {errorMsg}
        </div>
      )}
    </div>
  );
}
