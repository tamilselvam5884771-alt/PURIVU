import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import InteractionControls from './components/InteractionControls';
import AnswerCard from './components/AnswerCard';
import VisionUpload from './components/VisionUpload';
import VisionResult from './components/VisionResult';
import VoiceAssistant from './components/VoiceAssistant';
import PlaceholderModal from './components/PlaceholderModal';
import { chatQuery, analyzeProduct, checkHealth } from './services/api';
import { Search, Camera, Mic, ArrowRight, Loader2 } from 'lucide-react';

export default function App() {
  const [healthStatus, setHealthStatus] = useState(null);
  const [activeTab, setActiveTab] = useState('ask');
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [result, setResult] = useState(null);
  const [visionResult, setVisionResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [activeQuery, setActiveQuery] = useState('');
  const [error, setError] = useState('');
  const [activeModal, setActiveModal] = useState(null);

  useEffect(() => {
    fetchHealthStatus();
  }, []);

  const fetchHealthStatus = async () => {
    try {
      const data = await checkHealth();
      setHealthStatus(data);
    } catch (e) {
      setHealthStatus({ status: 'offline' });
    }
  };

  const handleVoiceQuery = async (recognizedText, lang) => {
    setLoading(true);
    setError('');
    setActiveQuery(recognizedText);
    setLoadingStep('Searching BIS RAG & synthesizing evidence in ' + lang + '...');

    try {
      const data = await chatQuery(recognizedText, lang);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to process voice query.');
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  };

  const handleTabSelect = (tab) => {
    setActiveTab(tab);
  };

  const handleTextSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError('');
    setActiveQuery(question);

    try {
      const data = await chatQuery(question);
      setResult(data);
    } catch (err) {
      setError(err.message || 'An error occurred connecting to PURIVU API.');
    } finally {
      setLoading(false);
    }
  };

  const handleVisionAnalyze = async (file, optionalQuestion, url) => {
    setLoading(true);
    setError('');
    setPreviewUrl(url);
    setLoadingStep('Examining product image & identifying features...');

    try {
      setTimeout(() => setLoadingStep('Understanding product type & category...'), 1000);
      setTimeout(() => setLoadingStep('Searching BIS RAG knowledge base for applicable standards...'), 2500);

      const data = await analyzeProduct(file, optionalQuestion);
      setVisionResult(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze product image.');
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  };

  const handleRequeryProductCorrection = async (newName, newCategory) => {
    setLoading(true);
    setError('');
    const query = `BIS Indian Standard certification requirements for ${newName} ${newCategory}`;
    setLoadingStep(`Searching BIS RAG for corrected product: ${newName}...`);

    try {
      const ragData = await chatQuery(query);
      setVisionResult({
        product: {
          name: newName,
          category: newCategory || 'User Corrected Category',
          description: `User corrected product identification to: ${newName}.`,
          confidence: 'High (User Corrected)'
        },
        answer: ragData.answer,
        evidence_status: ragData.evidence_status,
        sources: ragData.sources
      });
    } catch (err) {
      setError(err.message || 'Failed to search corrected product.');
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  };

  const handleSuggestionClick = (text) => {
    setQuestion(text);
    setActiveTab('ask');
  };

  const handleReset = () => {
    setResult(null);
    setVisionResult(null);
    setActiveQuery('');
    setQuestion('');
    setError('');
    setPreviewUrl(null);
  };

  const suggestions = [
    "What is hallmarking?",
    "How do I apply for BIS certification?",
    "What is a Certificate of Conformity?"
  ];

  return (
    <div className="min-h-screen flex flex-col justify-between text-[#29242A] bg-[#FFF9F5] selection:bg-[#F3C7B5] selection:text-[#4B1248]">
      {/* Header */}
      <Navbar healthStatus={healthStatus} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-6 py-6 flex flex-col justify-center">

        {/* Modal for Placeholder */}
        <PlaceholderModal
          type={activeModal}
          onClose={() => {
            setActiveModal(null);
            setActiveTab('ask');
          }}
        />

        {/* LANDING / INPUT STATE */}
        {!result && !visionResult && (
          <div className="space-y-8 my-auto py-6 animate-fade-in text-center">

            {/* Hero Section */}
            <div className="relative flex flex-col items-center justify-center my-2">
              {/* Subtle light theme knowledge pulse */}
              <div className="absolute w-44 h-44 rounded-full border border-[#4B1248]/10 animate-ring-1 pointer-events-none" />
              <div className="absolute w-60 h-60 rounded-full border border-[#D98268]/10 animate-ring-2 pointer-events-none" />
              <div className="absolute w-36 h-36 rounded-full bg-[#F3C7B5]/20 blur-2xl animate-knowledge-pulse pointer-events-none" />

              <h1 className="text-3xl sm:text-4xl font-semibold text-[#29242A] tracking-tight mb-1">
                PURIVU
              </h1>
              <p className="text-sm sm:text-base text-[#756873] font-normal max-w-md mx-auto">
                Understand standards. Make compliance simpler.
              </p>
            </div>

            {/* Capability Tabs (ASK, SHOW, VAANI) */}
            <InteractionControls
              activeTab={activeTab}
              onSelectTab={handleTabSelect}
            />

            {/* ERROR DISPLAY */}
            {error && (
              <div className="max-w-xl mx-auto p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs sm:text-sm">
                ⚠️ {error}
              </div>
            )}

            {/* ACTIVE MODE: ASK (Default Text Query) */}
            {activeTab === 'ask' && (
              <div className="space-y-6">
                <form onSubmit={handleTextSubmit} className="max-w-2xl mx-auto w-full">
                  <div className="bg-white rounded-2xl p-2 sm:p-2.5 flex items-center space-x-3 border border-[#EEDFD7] shadow-sm transition-all duration-200 focus-within:border-[#4B1248]">
                    <Search className="w-5 h-5 text-[#756873] shrink-0 ml-2.5" />
                    <input
                      type="text"
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="Ask about a BIS standard, certification or hallmarking..."
                      disabled={loading}
                      className="w-full bg-transparent border-0 outline-none text-[#29242A] placeholder-[#756873]/60 text-sm py-1.5"
                    />

                    <div className="flex items-center space-x-1.5 shrink-0">
                      <button
                        type="button"
                        onClick={() => handleTabSelect('vaani')}
                        title="VAANI Voice Assistant"
                        className="p-2 rounded-xl text-[#756873] hover:text-[#4B1248] hover:bg-[#FFF9F5] transition-colors cursor-pointer"
                      >
                        <Mic className="w-4 h-4" />
                      </button>
                      <button
                        type="button"
                        onClick={() => handleTabSelect('show')}
                        title="SHOW Vision Analysis"
                        className="p-2 rounded-xl text-[#756873] hover:text-[#4B1248] hover:bg-[#FFF9F5] transition-colors cursor-pointer"
                      >
                        <Camera className="w-4 h-4" />
                      </button>
                      <button
                        type="submit"
                        disabled={loading || !question.trim()}
                        className={`px-4 py-2 rounded-xl font-medium text-xs sm:text-sm transition-all cursor-pointer ${
                          question.trim() && !loading
                            ? 'bg-[#4B1248] hover:bg-[#64175F] text-white shadow-xs'
                            : 'bg-[#EEDFD7]/60 text-[#756873] cursor-not-allowed'
                        }`}
                      >
                        {loading ? (
                          <Loader2 className="w-4 h-4 animate-spin text-white" />
                        ) : (
                          <span>Ask</span>
                        )}
                      </button>
                    </div>
                  </div>
                </form>

                {/* Loading Sequence Indicator */}
                {loading && (
                  <div className="flex flex-col items-center justify-center space-y-3 py-4 animate-pulse">
                    <div className="w-8 h-8 rounded-full border-2 border-[#4B1248] border-t-transparent animate-spin text-[#2864E8]" />
                    <p className="text-xs text-[#756873] font-mono">
                      Checking BIS evidence & synthesizing answer...
                    </p>
                  </div>
                )}

                {/* Lightweight Example Suggestions */}
                {!loading && (
                  <div className="space-y-2 pt-1">
                    <div className="flex flex-wrap items-center justify-center gap-2 max-w-2xl mx-auto">
                      {suggestions.map((text, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleSuggestionClick(text)}
                          className="bg-white hover:bg-[#FBE3D5]/50 px-3.5 py-1.5 rounded-full text-xs text-[#756873] hover:text-[#4B1248] transition-colors border border-[#EEDFD7] shadow-2xs cursor-pointer"
                        >
                          "{text}"
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ACTIVE MODE: SHOW (Vision Upload) */}
            {activeTab === 'show' && (
              <div className="space-y-4">
                <VisionUpload onAnalyze={handleVisionAnalyze} loading={loading} />

                {loading && (
                  <div className="flex flex-col items-center justify-center space-y-3 py-4 animate-pulse">
                    <div className="w-8 h-8 rounded-full border-2 border-[#4B1248] border-t-transparent animate-spin" />
                    <p className="text-xs text-[#756873] font-mono">
                      {loadingStep || 'Analyzing product image & checking BIS evidence...'}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* ACTIVE MODE: VAANI (Voice Assistant) */}
            {activeTab === 'vaani' && (
              <div className="space-y-4">
                <VoiceAssistant onVoiceQuery={handleVoiceQuery} loading={loading} />
              </div>
            )}

          </div>
        )}

        {/* TEXT ANSWER RESULT STATE */}
        {result && (
          <div className="py-4">
            <AnswerCard
              query={activeQuery}
              result={result}
              onReset={handleReset}
            />
          </div>
        )}

        {/* VISION ANSWER RESULT STATE */}
        {visionResult && (
          <div className="py-4">
            <VisionResult
              visionData={visionResult}
              imagePreviewUrl={previewUrl}
              onRequeryProduct={handleRequeryProductCorrection}
              onReset={handleReset}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full py-4 text-center text-xs text-[#756873] border-t border-[#EEDFD7] mt-8 bg-white/50">
        PURIVU — BIS Intelligence
      </footer>
    </div>
  );
}
