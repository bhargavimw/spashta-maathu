import React from 'react';
import { Activity, Radio } from 'lucide-react';
import { HealthStatus } from '../types';

interface HeaderProps {
  health: HealthStatus | null;
}

export const Header: React.FC<HeaderProps> = ({ health }) => {
  const ready = health?.components.vad ?? false;

  return (
    <header className="border-b border-slate-900 bg-slate-900/60 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-2xl gradient-text flex items-center justify-center text-white font-bold shadow-lg shadow-indigo-500/25">
            <Radio className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-white tracking-tight">
                ಕನ್ನಡ <span className="gradient-text ml-1">Pause Remover</span>
              </h1>
              <span className="px-2 py-0.5 rounded-full font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 text-xs">
                v1 Pipeline
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Deterministic audio pause removal • VAD segmentation • zero-crossing splicing
            </p>
          </div>
        </div>

        {health && (
          <div className="flex items-center gap-3 max-w-xs w-full">
            <div className="flex-1 bg-slate-800/60 border border-slate-700/60 rounded-xl px-3 py-1.5 overflow-hidden">
              <div className="flex items-center justify-between text-xs text-slate-300">
                <span className="flex items-center gap-1.5">
                  <Activity className="w-3 h-3 text-emerald-400" />
                  <span>API {health.status}</span>
                </span>
                <span className={ready ? 'text-emerald-400' : 'text-slate-600'}>{ready ? 'online' : 'limited'}</span>
              </div>
              <div className="flex flex-wrap gap-2 mt-1.5">
                <span
                  className={`px-2 py-0.5 rounded-md text-xs font-medium border ${health.components.vad ? 'badge-ready' : 'badge-off'}`}
                >
                  VAD
                </span>
                <span
                  className={`px-2 py-0.5 rounded-md text-xs font-medium border ${health.components.asr ? 'badge-ready' : 'badge-off'}`}
                  title="ASR deferred to M3"
                >
                  ASR (M3)
                </span>
                <span
                  className={`px-2 py-0.5 rounded-md text-xs font-medium border ${health.components.classifier ? 'badge-ready' : 'badge-off'}`}
                  title="LLM classifier deferred to M4"
                >
                  Classifier (M4)
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};
