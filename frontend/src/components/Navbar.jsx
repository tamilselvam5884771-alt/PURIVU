import React from 'react';

export default function Navbar({ healthStatus }) {
  const isHealthy = healthStatus?.status === 'ok';

  return (
    <header className="sticky top-0 z-40 w-full bg-white/90 backdrop-blur-md border-b border-[#EEDFD7] px-4 sm:px-8 py-3.5 mb-6 shadow-xs">
      <div className="max-w-5xl mx-auto flex items-center justify-between">
        {/* Official Brand Logo & Name */}
        <div className="flex items-center space-x-3">
          <img
            src="/purivu-logo.png"
            alt="PURIVU Logo"
            className="h-9 w-auto rounded-lg shadow-sm"
          />
          <div className="flex items-center space-x-2">
            <span className="font-heading font-extrabold text-xl text-[#4B1248] tracking-tight">
              PURIVU
            </span>
            <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-[#FBE3D5] text-[#4B1248] border border-[#EEDFD7]">
              BIS Intelligence
            </span>
          </div>
        </div>

        {/* Health Status Indicator */}
        <div className="flex items-center space-x-2 bg-[#FFF9F5] px-3 py-1.5 rounded-full border border-[#EEDFD7] text-xs">
          <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
          <span className="text-[#4B1248] font-medium hidden sm:inline">
            {isHealthy ? 'RAG Ready' : 'Connecting...'}
          </span>
          {healthStatus?.total_chunks && (
            <span className="text-[11px] text-[#756873] font-mono hidden md:inline">
              ({healthStatus.total_chunks} Chunks)
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
