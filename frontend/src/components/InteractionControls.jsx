import React from 'react';
import { MessageSquareText, Camera, Mic } from 'lucide-react';

export default function InteractionControls({ activeTab, onSelectTab }) {
  const modes = [
    {
      id: 'ask',
      label: 'ASK',
      desc: 'Text Query',
      icon: MessageSquareText
    },
    {
      id: 'show',
      label: 'SHOW',
      desc: 'Product Vision',
      icon: Camera
    },
    {
      id: 'vaani',
      label: 'VAANI',
      desc: 'Voice Assistant',
      icon: Mic
    }
  ];

  return (
    <div className="flex items-center justify-center p-1.5 rounded-2xl bg-[#FFF9F5] border border-[#EEDFD7] max-w-md mx-auto mb-6 shadow-sm">
      {modes.map((mode) => {
        const Icon = mode.icon;
        const isActive = mode.id === activeTab;
        return (
          <button
            key={mode.id}
            onClick={() => onSelectTab(mode.id)}
            className={`flex-1 flex items-center justify-center space-x-2 py-2.5 px-4 rounded-xl text-xs sm:text-sm font-semibold transition-all duration-250 cursor-pointer ${
              isActive
                ? 'bg-[#4B1248] text-white shadow-md shadow-[#4B1248]/20 scale-[1.02]'
                : 'text-[#756873] hover:text-[#4B1248] hover:bg-[#FBE3D5]/50'
            }`}
          >
            <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-[#D98268]'}`} />
            <span>{mode.label}</span>
          </button>
        );
      })}
    </div>
  );
}
