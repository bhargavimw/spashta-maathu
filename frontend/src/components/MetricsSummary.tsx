import React from 'react';
import { Download, Clock, Scissors, TrendingDown } from 'lucide-react';
import { ProcessResult } from '../types';

interface MetricsSummaryProps {
  result: ProcessResult;
}

const METRICS = [
  {
    label: 'Original',
    valueKey: 'original_duration',
    color: 'text-slate-300',
    icon: Clock,
    suffix: 's',
  },
  {
    label: 'Cleaned',
    valueKey: 'clean_duration',
    color: 'text-emerald-400',
    icon: Clock,
    suffix: 's',
    accent: true,
  },
  {
    label: 'Time Saved',
    valueKey: 'time_saved',
    color: 'text-indigo-400',
    icon: TrendingDown,
    suffix: 's',
  },
  {
    label: 'Pauses Capped',
    valueKey: 'removed_regions.length',
    color: 'text-rose-400',
    icon: Scissors,
    suffix: '',
    count: true,
  },
] as const;

export const MetricsSummary: React.FC<MetricsSummaryProps> = ({ result }) => {
  const percentSaved =
    result.original_duration > 0
      ? ((result.time_saved / result.original_duration) * 100).toFixed(1)
      : '0.0';

  return (
    <section className="fade-in slide-up" style={{ animationDelay: '0.1s' }}>
      <div className="glass p-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/10 rounded-full blur-3xl -mr-6 -mt-6" />

        <div className="relative z-10">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center">
                <BarChartIcon />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white tracking-tight">Reduction Metrics</h2>
                <p className="text-xs text-slate-400">
                  Silence trimming and gap-capping summary for this clip
                </p>
              </div>
            </div>

            <a
              href={result.clean_audio_url}
              download="kannada_clean.wav"
              className="btn-primary text-xs px-3 py-2"
            >
              <Download className="w-3 h-3 mr-2" />
              Download .wav
            </a>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {METRICS.map((m) => {
              let value: React.ReactNode;
              if (m.count) {
                value = result.removed_regions.length;
              } else if (m.valueKey === 'time_saved') {
                value = result.time_saved.toFixed(2);
              } else {
                value = result[m.valueKey as keyof ProcessResult].toFixed(2);
              }

              return (
                <div
                  key={m.label}
                  className={`rounded-2xl p-4 border transition-all hover:border-indigo-500/40 ${
                    m.accent
                      ? 'bg-emerald-500/5 border-emerald-500/20 shadow-lg shadow-emerald-500/10'
                      : 'bg-slate-950/60 border-slate-800/80'
                  }`}
                >
                  <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1.5">
                    <m.icon className={`w-3.5 h-3.5 ${m.color}`} />
                    <span>{m.label}</span>
                  </div>
                  <div className="flex items-end gap-1">
                    <p className={`text-2xl font-bold tracking-tight ${m.color}`}>{value}</p>
                    <span className="text-xs font-normal text-slate-500 mb-0.5">{m.suffix}</span>
                  </div>
                  {m.label === 'Time Saved' && (
                    <p className="text-xs text-slate-500 mt-1">{percentSaved}% reduction</p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

// Small inline icon to avoid an extra import file
function BarChartIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="12" y1="20" x2="12" y2="20" />
      <line x1="18" y1="20" x2="18" y2="20" />
      <line x1="6" y1="20" x2="6" y2="20" />
      <line x1="12" y1="14" x2="12" y2="20" />
      <line x1="18" y1="8" x2="18" y2="20" />
      <line x1="6" y1="16" x2="6" y2="20" />
    </svg>
  );
}
