import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Mic, Square, Play, Pause, RefreshCw, FileAudio, MicOff } from 'lucide-react';

interface AudioUploaderProps {
  onProcess: (file: File) => Promise<void>;
  isLoading: boolean;
}

export const AudioUploader: React.FC<AudioUploaderProps> = ({ onProcess, isLoading }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [isPlayingPreview, setIsPlayingPreview] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const audioPreviewRef = useRef<HTMLAudioElement | null>(null);

  // Clean up object URLs
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [audioUrl]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setAudio(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setAudio(e.dataTransfer.files[0]);
    }
  };

  const setAudio = (file: File) => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setIsPlayingPreview(false);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const recordedFile = new File([audioBlob], `recording_${Date.now()}.wav`, {
          type: 'audio/wav',
        });
        setAudio(recordedFile);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordSeconds(0);

      timerRef.current = window.setInterval(() => {
        setRecordSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Microphone access denied:', err);
      alert('Unable to access microphone. Please ensure microphone permissions are granted.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const togglePreview = () => {
    if (!audioPreviewRef.current) return;
    if (isPlayingPreview) {
      audioPreviewRef.current.pause();
      setIsPlayingPreview(false);
    } else {
      audioPreviewRef.current.play();
      setIsPlayingPreview(true);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile || isLoading) return;
    await onProcess(selectedFile);
  };

  const formatSeconds = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const rem = sec % 60;
    return `${mins.toString().padStart(2, '0')}:${rem.toString().padStart(2, '0')}`;
  };

  return (
    <section className="fade-in slide-up" style={{ animationDelay: '0.1s' }}>
      <div className="glass p-6 relative overflow-hidden">
        {/* Decorative accent corner */}
        <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/10 rounded-full blur-3xl -mr-6 -mt-6" />

        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-8 h-8 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center">
              <FileAudio className="w-5 h-5" />
            </div>
            <h2 className="text-lg font-semibold text-white tracking-tight">
              Input Audio
            </h2>
          </div>

          <p className="text-sm text-slate-400 mb-6">
            Upload a hesitant Kannada clip or record live. We’ll trim silence and cap pauses.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6">
            {/* Upload zone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDragLeave={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className="relative group border-2 border-dashed border-slate-700 rounded-2xl p-8 text-center cursor-pointer transition-all hover:bg-slate-900/40 hover:border-indigo-500/50"
            >
              <input
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                disabled={isLoading || isRecording}
              />
              <div className="relative z-10 flex flex-col items-center gap-2">
                <UploadCloud className="w-10 h-10 text-slate-400 group-hover:text-indigo-400 transition-colors" />
                <p className="text-sm font-medium text-slate-200 group-hover:text-white transition-colors">
                  Drop file here, or <span className="text-indigo-400 underline underline-offset-2">browse</span>
                </p>
                <p className="text-xs text-slate-500 mt-0.5">WAV • MP3 • M4A • FLAC • OGG</p>
              </div>
            </div>

            {/* Live recorder */}
            <div className="border border-slate-800 rounded-2xl p-6 bg-slate-950/40 flex flex-col items-center justify-center">
              {isRecording ? (
                <div className="flex flex-col items-center gap-3">
                  <div className="flex items-center gap-2 text-rose-400 font-mono text-xl font-semibold">
                    <span className="w-3 h-3 rounded-full bg-rose-500 recording-pulse" />
                    {formatSeconds(recordSeconds)}
                  </div>
                  <p className="text-xs text-slate-400">Speak with natural pauses...</p>
                  <button
                    onClick={stopRecording}
                    className="mt-2 px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-sm font-medium shadow-lg shadow-rose-500/25 transition-all"
                  >
                    <Square className="w-4 h-4" /> Stop
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2">
                  <button
                    onClick={startRecording}
                    disabled={isLoading}
                    className={`p-4 rounded-full transition-all ${
                      isLoading
                        ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                        : 'bg-indigo-600/10 hover:bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 hover:border-indigo-500/50 group'
                    }`}
                  >
                    <Mic className="w-7 h-7 group-hover:scale-110 transition-transform" />
                  </button>
                  <p className="text-sm font-medium text-slate-200">Record live</p>
                  <p className="text-xs text-slate-500">In-browser capture</p>
                </div>
              )}
            </div>
          </div>

          {/* Selected file preview bar */}
          {selectedFile && (
            <div className="mt-5 pt-5 border-t border-slate-800/50 bg-slate-950/40 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 animate-in fade-in">
              <div className="flex items-center gap-3">
                <button
                  onClick={togglePreview}
                  className="p-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
                >
                  {isPlayingPreview ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                <div className="flex flex-col">
                  <p className="text-sm font-medium text-white truncate max-w-xs sm:max-w-md">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-slate-400">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • {selectedFile.type || 'audio/wav'}
                  </p>
                </div>
                <audio
                  ref={audioPreviewRef}
                  src={audioUrl}
                  onEnded={() => setIsPlayingPreview(false)}
                  className="hidden"
                />
              </div>

              <button
                onClick={handleSubmit}
                disabled={isLoading}
                className="btn-primary whitespace-nowrap"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Processing…
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" /> Clean Audio
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
