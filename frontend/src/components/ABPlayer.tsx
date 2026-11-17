import React, { useState } from 'react';
import { ToggleLeft, ToggleRight, Sparkles, Volume2 } from 'lucide-react';
import { WaveformViewer } from './WaveformViewer';
import { ProcessResult } from '../types';

interface ABPlayerProps {
  originalAudioUrl: string;
  result: ProcessResult;
}

export const ABPlayer: React.FC<ABPlayerProps> = ({ originalAudioUrl, result }) => {
  // 'A' = Original, 'B' = Cleaned
  const [activeTrack, setActiveTrack] = useState<'A' | 'B'>('B');

  const toggleTrack = () => {
    setActiveTrack((prev) => (prev === 'A' ? 'B' : 'A'));
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <div>
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Volume2 className="w-5 h-5 text-indigo-400" />
            A/B Listening & Comparison
          </h2>
          <p className="text-xs text-slate-400">
            Toggle between Track A (original hesitant audio) and Track B (cleaned audio)
          </p>
        </div>

        {/* Master A/B Switch Button */}
        <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 p-1.5 rounded-xl">
          <button
            onClick={() => setActiveTrack('A')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
              activeTrack === 'A'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <span>Track A: Original</span>
            {activeTrack === 'A' && <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />}
          </button>

          <button
            onClick={toggleTrack}
            className="p-1 text-slate-400 hover:text-white transition-colors"
            title="Toggle A/B"
          >
            {activeTrack === 'A' ? (
              <ToggleLeft className="w-6 h-6 text-amber-400" />
            ) : (
              <ToggleRight className="w-6 h-6 text-emerald-400" />
            )}
          </button>

          <button
            onClick={() => setActiveTrack('B')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
              activeTrack === 'B'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            <span>Track B: Cleaned</span>
            {activeTrack === 'B' && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
          </button>
        </div>
      </div>

      {/* Comparative Waveforms Display */}
      <div className="space-y-4">
        {/* Track A: Original Waveform */}
        <div
          className={`transition-all duration-200 rounded-2xl ${
            activeTrack === 'A'
              ? 'ring-2 ring-amber-500/50 shadow-lg shadow-amber-500/10'
              : 'opacity-70 hover:opacity-100'
          }`}
        >
          <WaveformViewer
            audioUrl={originalAudioUrl}
            removedRegions={result.removed_regions}
            title="Track A: Original Audio (Red overlays = Cut pauses)"
            waveColor="#475569"
            progressColor="#f59e0b"
          />
        </div>

        {/* Track B: Cleaned Waveform */}
        <div
          className={`transition-all duration-200 rounded-2xl ${
            activeTrack === 'B'
              ? 'ring-2 ring-emerald-500/50 shadow-lg shadow-emerald-500/10'
              : 'opacity-70 hover:opacity-100'
          }`}
        >
          <WaveformViewer
            audioUrl={result.clean_audio_url}
            title="Track B: Cleaned Audio (Smoothed continuous speech)"
            waveColor="#334155"
            progressColor="#10b981"
          />
        </div>
      </div>
    </div>
  );
};
