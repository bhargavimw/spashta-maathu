import React from 'react';
import { Download, Clock, Scissors, TrendingDown } from 'lucide-react';
import { ProcessResult } from '../types';

interface MetricsSummaryProps {
  result: ProcessResult;
}

export const MetricsSummary: React.FC<MetricsSummaryProps> = ({ result }) => {
  const percentSaved =
    result.original_duration > 0
      ? ((result.time_saved / result.original_duration) * 100).toFixed(1)
      : '0.0';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <div>
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <TrendingDown className="w-5 h-5 text-indigo-400" />
            Pause Reduction Metrics
          </h2>
          <p className="text-xs text-slate-400">
            Summary of silence trimming and gap-capping operations
          </p>
        </div>

        <a
          href={result.clean_audio_url}
          download="kannada_clean.wav"
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-semibold flex items-center gap-2 shadow-lg shadow-emerald-600/20 transition-all"
        >
          <Download className="w-4 h-4" />
          Download Clean Audio (.wav)
        </a>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {/* Original Duration */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
            <Clock className="w-3.5 h-3.5 text-slate-400" />
            <span>Original</span>
          </div>
          <p className="text-2xl font-bold text-white tracking-tight">
            {result.original_duration.toFixed(2)}
            <span className="text-sm font-normal text-slate-400 ml-1">s</span>
          </p>
        </div>

        {/* Clean Duration */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
          <div className="flex items-center gap-1.5 text-xs text-emerald-400 mb-1">
            <Clock className="w-3.5 h-3.5 text-emerald-400" />
            <span>Cleaned</span>
          </div>
          <p className="text-2xl font-bold text-emerald-400 tracking-tight">
            {result.clean_duration.toFixed(2)}
            <span className="text-sm font-normal text-emerald-500 ml-1">s</span>
          </p>
        </div>

        {/* Time Saved */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
          <div className="flex items-center gap-1.5 text-xs text-indigo-400 mb-1">
            <TrendingDown className="w-3.5 h-3.5 text-indigo-400" />
            <span>Time Saved</span>
          </div>
          <p className="text-2xl font-bold text-indigo-400 tracking-tight">
            {result.time_saved.toFixed(2)}
            <span className="text-sm font-normal text-indigo-500 ml-1">s</span>
          </p>
          <p className="text-xs text-slate-500 mt-0.5">{percentSaved}% reduction</p>
        </div>

        {/* Removed Regions Count */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
          <div className="flex items-center gap-1.5 text-xs text-rose-400 mb-1">
            <Scissors className="w-3.5 h-3.5 text-rose-400" />
            <span>Pauses Capped</span>
          </div>
          <p className="text-2xl font-bold text-rose-400 tracking-tight">
            {result.removed_regions.length}
          </p>
          <p className="text-xs text-slate-500 mt-0.5">zero-crossing cuts</p>
        </div>
      </div>
    </div>
  );
};
