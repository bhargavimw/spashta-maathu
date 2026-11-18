import React from 'react';
import { FileText, Info, CheckCircle2 } from 'lucide-react';
import { WordToken } from '../types';

interface TranscriptDiffPanelProps {
  words: WordToken[] | null;
  verbatimText: string | null;
  cleanText: string | null;
  diff: unknown | null;
}

export const TranscriptDiffPanel: React.FC<TranscriptDiffPanelProps> = ({
  words,
  verbatimText,
  cleanText,
  diff,
}) => {
  const hasTranscript = Boolean(cleanText || (words && words.length > 0));
  const diffList = diff as Array<{ index: number; text: string; action: string; reason?: string }> | null;

  return (
    <section className="fade-in slide-up" style={{ animationDelay: '0.3s' }}>
      <div className="glass p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
            <FileText className="w-5 h-5 text-indigo-400" />
            Transcript &amp; Diff
          </h2>
          {hasTranscript ? (
            <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs flex items-center gap-1.5 font-medium">
              <CheckCircle2 className="w-3 h-3" />
              Text available
            </span>
          ) : (
            <span className="px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700/60 text-xs font-medium">
              Stage 2/4 pending
            </span>
          )}
        </div>

        {!hasTranscript ? (
          <div className="border border-dashed border-slate-800 rounded-2xl p-8 text-center bg-slate-950/40">
            <div className="inline-flex p-3 rounded-full bg-slate-800/80 text-slate-400 mb-4">
              <Info className="w-6 h-6" />
            </div>
            <h3 className="text-base font-medium text-slate-200 mb-1">
              Transcript not available yet
            </h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
              Automatic Speech Recognition (Stage 2) and LLM Disfluency Classification (Stage 4) are scheduled
              for Milestones M3–M5. Currently operating under the v1 pause remover pipeline.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {verbatimText && (
              <div className="bg-slate-950/60 border border-slate-800/80 rounded-2xl p-5">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                    Verbatim Transcript
                  </span>
                  <span className="ml-auto text-[10px] text-slate-500">{verbatimText.split(/\s+/).length} words</span>
                </div>
                <p className="text-sm text-slate-200 leading-relaxed font-sans italic">
                  {verbatimText}
                </p>
              </div>
            )}

            {cleanText && (
              <div className="bg-emerald-950/20 border border-emerald-800/40 rounded-2xl p-5 shadow-lg shadow-emerald-500/10">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400 block">
                    Cleaned Transcript
                  </span>
                  <span className="ml-auto text-[10px] text-emerald-500">{cleanText.split(/\s+/).length} words</span>
                </div>
                <p className="text-sm text-emerald-100 leading-relaxed font-sans font-medium">
                  {cleanText}
                </p>
              </div>
            )}
          </div>
        )}

        {diffList && diffList.length > 0 && (
          <div className="mt-6 border border-slate-800/50 bg-slate-950/30 rounded-2xl p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
              Segment-Level Diff
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {diffList.map((entry) => (
                <div
                  key={entry.index}
                  className={`rounded-lg p-2.5 border text-xs ${
                    entry.action === 'keep'
                      ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-100'
                      : 'bg-rose-500/5 border-rose-500/20 text-rose-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={`font-bold ${entry.action === 'keep' ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {entry.action === 'keep' ? '✓ keep' : '✗ remove'}
                    </span>
                    {entry.reason && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-current/10 text-current">
                        {entry.reason}
                      </span>
                    )}
                  </div>
                  <p className="text-slate-300 truncate" title={entry.text}>
                    [{entry.index}] {entry.text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
