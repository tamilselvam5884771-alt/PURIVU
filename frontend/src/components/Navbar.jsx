import React from 'react';

export default function Navbar({ healthStatus }) {
  const isHealthy = healthStatus?.status === 'ok';

  return (
    <header className="sticky top-0 z-40 w-full bg-[#FFFDFC]/95 backdrop-blur-md border-b border-[#E8C5B5] px-4 sm:px-8 py-3.5 mb-6 shadow-xs">
      <div className="max-w-5xl mx-auto flex items-center justify-between">
        {/* Official Brand Logo & Name */}
        <div className="flex items-center space-x-3">
          <img
            src="/purivu-logo.png"
            alt="PURIVU Logo"
            className="h-10 w-10 rounded-full object-cover shadow-xs"
          />
          <div className="flex items-center space-x-2">
            <span className="font-heading font-extrabold text-xl text-[#5A1855] tracking-tight">
              PURIVU
            </span>
            <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-[#FBE0D2] text-[#6A2365] border border-[#E7BDAA]">
              BIS Intelligence
            </span>
          </div>
        </div>

        {/* Health Status Indicator */}
        <div className="flex items-center space-x-2 bg-[#FFF9F5] px-3.5 py-1.5 rounded-full border border-[#E2B6A3] text-xs shadow-2xs">
          <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
          <span className="text-[#5A1855] font-semibold hidden sm:inline">
            {isHealthy ? 'RAG Ready' : 'Connecting...'}
          </span>
          {healthStatus?.total_chunks && (
            <span className="text-[11px] text-[#735F6C] font-mono font-medium hidden md:inline">
              ({healthStatus.total_chunks} Chunks)
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
