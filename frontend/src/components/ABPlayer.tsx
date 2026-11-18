import React, { useState } from 'react';
import { ToggleLeft, ToggleRight, Sparkles, Volume2 } from 'lucide-react';
import { WaveformViewer } from './WaveformViewer';
import { ProcessResult } from '../types';

interface ABPlayerProps {
  originalAudioUrl: string;
  result: ProcessResult;
}

export const ABPlayer: React.FC<ABPlayerProps> = ({ originalAudioUrl, result }) => {
  const [activeTrack, setActiveTrack] = useState<'A' | 'B'>('B');

  return (
    <section className="fade-in slide-up" style={{ animationDelay: '0.2s' }}>
      <div className="glass p-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-3xl -mr-8 -mt-8" />

        <div className="relative z-10">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center">
                <Volume2 className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white tracking-tight">A/B Listening</h2>
                <p className="text-xs text-slate-400">
                  Toggle between the original hesitant audio and the cleaned result
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 bg-slate-900 border border-slate-700 p-1 rounded-2xl">
              <button
                onClick={() => setActiveTrack('A')}
                className={`px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
                  activeTrack === 'A'
                    ? 'bg-amber-900/30 text-amber-300 border border-amber-500/40 shadow-lg shadow-amber-500/10'
                    : 'text-slate-400 hover:text-slate-200 hover:border-slate-600'
                }`}
              >
                <span>Track A</span>
                {activeTrack === 'A' && <span className="w-2 h-2 rounded-full bg-amber-400" />}
              </button>

              <button
                onClick={() => setActiveTrack('B')}
                className={`px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
                  activeTrack === 'B'
                    ? 'bg-emerald-900/30 text-emerald-300 border border-emerald-500/40 shadow-lg shadow-emerald-500/10'
                    : 'text-slate-400 hover:text-slate-200 hover:border-slate-600'
                }`}
              >
                <Sparkles className="w-3 h-3" />
                <span>Track B</span>
                {activeTrack === 'B' && <span className="w-2 h-2 rounded-full bg-emerald-400" />}
              </button>
            </div>
          </div>

          <div className="space-y-4">
            <div
              className={`rounded-2xl p-1 border transition-all duration-300 ${
                activeTrack === 'A'
                  ? 'ring-2 ring-amber-500/50 bg-amber-500/5 shadow-lg shadow-amber-500/10 border-amber-500/30'
                  : 'border-slate-800/50 opacity-90 hover:opacity-100'
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-medium text-amber-400 flex items-center gap-1">
                  <ToggleLeft className="w-3 h-3" /> Original
                </span>
                <span className="text-xs text-slate-500">Hesitant input</span>
              </div>
              <WaveformViewer
                audioUrl={originalAudioUrl}
                removedRegions={result.removed_regions}
                title="Original audio — red overlays show removed pauses"
                waveColor="#475569"
                progressColor="#f59e0b"
              />
            </div>

            <div
              className={`rounded-2xl p-1 border transition-all duration-300 ${
                activeTrack === 'B'
                  ? 'ring-2 ring-emerald-500/50 bg-emerald-500/5 shadow-lg shadow-emerald-500/10 border-emerald-500/30'
                  : 'border-slate-800/50 opacity-90 hover:opacity-100'
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-medium text-emerald-400 flex items-center gap-1">
                  <Sparkles className="w-3 h-3" /> Cleaned
                </span>
                <span className="text-xs text-slate-500">Smoothed output</span>
              </div>
              <WaveformViewer
                audioUrl={result.clean_audio_url}
                title="Cleaned audio — continuous speech"
                waveColor="#334155"
                progressColor="#10b981"
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
