import React, { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { AudioUploader } from './components/AudioUploader';
import { ABPlayer } from './components/ABPlayer';
import { MetricsSummary } from './components/MetricsSummary';
import { TranscriptDiffPanel } from './components/TranscriptDiffPanel';
import { HealthStatus, ProcessResult } from './types';
import { AlertCircle } from 'lucide-react';

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

    // Keep client-side preview URL for the original audio
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

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col font-sans">
      <Header health={health} />

      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 space-y-8">
        {/* Error notification banner */}
        {error && (
          <div className="bg-rose-500/10 border border-rose-500/30 text-rose-300 p-4 rounded-xl flex items-center gap-3 text-sm">
            <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        {/* 1. File Upload & Microphone Audio Recorder */}
        <section>
          <AudioUploader onProcess={handleProcessAudio} isLoading={isLoading} />
        </section>

        {/* 2. Processing Results Section */}
        {result && originalAudioUrl && (
          <section className="space-y-8 animate-in fade-in duration-300">
            {/* Reduction Metrics */}
            <MetricsSummary result={result} />

            {/* A/B Listening Player with Waveforms and Red Overlays */}
            <ABPlayer originalAudioUrl={originalAudioUrl} result={result} />

            {/* Transcript & Text Diff (Neutral informative state for v1) */}
            <TranscriptDiffPanel
              words={result.words}
              verbatimText={result.verbatim_text}
              cleanText={result.clean_text}
              diff={result.diff}
            />
          </section>
        )}
      </main>

      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-500">
        Kannada Pause Remover & Stammer Cleaner • Milestone M9 React + Wavesurfer.js Interface
      </footer>
    </div>
  );
};

export default App;
