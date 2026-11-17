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
}) => {
  const hasTranscript = Boolean(cleanText || (words && words.length > 0));

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <FileText className="w-5 h-5 text-indigo-400" />
          Transcript & Text Diff
        </h2>
        {hasTranscript ? (
          <span className="text-xs px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5 font-medium">
            <CheckCircle2 className="w-3.5 h-3.5" />
            ASR & LLM Text Available
          </span>
        ) : (
          <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700/60 font-medium">
            Stage 2/4 Pending
          </span>
        )}
      </div>

      {!hasTranscript ? (
        /* Neutral "Transcript not available yet" state per requirements */
        <div className="border border-dashed border-slate-800 rounded-xl p-8 text-center bg-slate-950/40">
          <div className="inline-flex p-3 rounded-full bg-slate-800/80 text-slate-400 mb-3">
            <Info className="w-6 h-6" />
          </div>
          <h3 className="text-base font-medium text-slate-200 mb-1">
            Transcript not available yet
          </h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
            Automatic Speech Recognition (Stage 2) and LLM Disfluency Classification (Stage 4) are scheduled for Milestones M3–M5. Currently operating under the v1 pause remover pipeline.
          </p>
        </div>
      ) : (
        /* Real transcript display when present */
        <div className="space-y-4">
          {verbatimText && (
            <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
                Verbatim Transcript
              </span>
              <p className="text-sm text-slate-200 leading-relaxed font-sans">
                {verbatimText}
              </p>
            </div>
          )}

          {cleanText && (
            <div className="bg-emerald-950/20 border border-emerald-800/40 rounded-xl p-4">
              <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400 block mb-2">
                Cleaned Transcript
              </span>
              <p className="text-sm text-emerald-100 leading-relaxed font-sans font-medium">
                {cleanText}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
