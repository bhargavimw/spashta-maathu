import React from 'react';
import { Activity, Radio } from 'lucide-react';
import { HealthStatus } from '../types';

interface HeaderProps {
  health: HealthStatus | null;
}

export const Header: React.FC<HeaderProps> = ({ health }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50 px-6 py-4">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
            <Radio className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-white tracking-tight">
                ಕನ್ನಡ Pause Remover & Stammer Cleaner
              </h1>
              <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                v1 Pipeline
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Deterministic audio pause removal, VAD segmentation & zero-crossing splicer
            </p>
          </div>
        </div>

        {health && (
          <div className="flex items-center gap-2 text-xs bg-slate-800/80 border border-slate-700/60 rounded-lg px-3 py-1.5">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-300 font-medium">API: {health.status}</span>
            <span className="text-slate-500">|</span>
            <div className="flex items-center gap-1.5 text-slate-400">
              <span className={health.stages.ingest ? 'text-emerald-400' : 'text-slate-600'}>Ingest</span>
              <span>•</span>
              <span className={health.stages.vad ? 'text-emerald-400' : 'text-slate-600'}>VAD</span>
              <span>•</span>
              <span className={health.stages.splicer ? 'text-emerald-400' : 'text-slate-600'}>Splicer</span>
              <span>•</span>
              <span className={health.stages.asr ? 'text-emerald-400' : 'text-slate-600'} title="ASR deferred to M3">ASR (M3)</span>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};
