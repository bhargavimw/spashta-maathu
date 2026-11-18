import React, { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { AudioUploader } from './components/AudioUploader';
import { ABPlayer } from './components/ABPlayer';
import { MetricsSummary } from './components/MetricsSummary';
import { TranscriptDiffPanel } from './components/TranscriptDiffPanel';
import { HealthStatus, ProcessResult } from './types';
import { AlertCircle, BarChart3, Wand2 } from 'lucide-react';

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [originalAudioUrl, setOriginalAudioUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ProcessResult | null>(null);

  // Check health on mount
  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch (err) {
      console.warn('Backend API /health unreachable on startup:', err);
    }
  };

  const handleProcessAudio = async (file: File) => {
    setIsLoading(true);
    setError(null);

    if (originalAudioUrl) {
      URL.revokeObjectURL(originalAudioUrl);
    }
    const origUrl = URL.createObjectURL(file);
    setOriginalAudioUrl(origUrl);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: 'Processing error' }));
        throw new Error(errData.detail || `Server responded with ${response.status}`);
      }

      const data: ProcessResult = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error('Audio processing failed:', err);
      setError(err.message || 'Failed to process audio. Ensure backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const hasResult = Boolean(result && originalAudioUrl);

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col font-sans antialiased">
      <Header health={health} />

      <main className="flex-1 page">
        {error && (
          <div className="bg-rose-900/20 border border-rose-500/30 text-rose-300 p-5 rounded-2xl flex items-start gap-3 mb-6 animate-in fade-in">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-rose-400" />
            <div>
              <p className="font-medium text-sm">{error}</p>
              <p className="text-xs text-rose-400/70 mt-1">Try uploading a valid WAV/MP3 file.</p>
            </div>
          </div>
        )}

        {/* Always-visible upload & recorder */}
        <AudioUploader onProcess={handleProcessAudio} isLoading={isLoading} />

        {!hasResult ? (
          <section className="empty-hero fade-in">
            <span className="emoji">🎙️</span>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Kannada <span className="gradient-text">Pause Remover</span>{' & '}Stammer Cleaner
            </h1>
            <p>
              Upload hesitant Kannada speech and get a smoothed, clean audio file with
              unnatural pauses, stammers, and fillers removed — preserving the speaker's
              meaning and natural prosody.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 justify-center items-center">
              <span className="px-4 py-1.5 rounded-xl bg-slate-900/50 border border-slate-800 text-xs text-slate-400 flex items-center gap-2">
                <BarChart3 className="w-3 h-3 text-slate-500" /> V1 pipeline active
              </span>
              <span className="px-4 py-1.5 rounded-xl bg-slate-900/50 border border-slate-800 text-xs text-slate-400 flex items-center gap-2">
                <Wand2 className="w-3 h-3 text-slate-500" />{' ASR & LLM coming in M3–M5'}
              </span>
            </div>
          </section>
        ) : (
          <section className="result-grid">
            {/* Left column: player */}
            <div className="space-y-6">
              <ABPlayer originalAudioUrl={originalAudioUrl} result={result} />
            </div>

            {/* Right column: metrics + transcript */}
            <div className="space-y-6">
              <MetricsSummary result={result} />
              <TranscriptDiffPanel
                words={result.words}
                verbatimText={result.verbatim_text}
                cleanText={result.clean_text}
                diff={result.diff}
              />
            </div>
          </section>
        )}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/80 py-5 text-center text-xs text-slate-500 backdrop-blur">
        Kannada Pause Remover{' & '}Stammer Cleaner • V1 pause remover pipeline
      </footer>
    </div>
  );
};

export default App;
