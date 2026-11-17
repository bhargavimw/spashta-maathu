import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Mic, Square, Play, Pause, RefreshCw, FileAudio } from 'lucide-react';

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
      const file = e.target.files[0];
      setAudio(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setAudio(file);
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
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <FileAudio className="w-5 h-5 text-indigo-400" />
        Input Audio (Upload or Record)
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Upload Zone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className="border-2 border-dashed border-slate-700 hover:border-indigo-500/60 rounded-xl p-6 text-center cursor-pointer transition-colors bg-slate-950/40 hover:bg-slate-950/70 flex flex-col items-center justify-center min-h-[160px] relative"
        >
          <input
            type="file"
            accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg"
            onChange={handleFileChange}
            className="absolute inset-0 opacity-0 cursor-pointer"
            disabled={isLoading || isRecording}
          />
          <UploadCloud className="w-10 h-10 text-slate-400 mb-2" />
          <p className="text-sm font-medium text-slate-200">
            Drop hesitant audio file here, or <span className="text-indigo-400 underline">browse</span>
          </p>
          <p className="text-xs text-slate-500 mt-1">Supports WAV, MP3, M4A, FLAC, OGG</p>
        </div>

        {/* Live Audio Recorder */}
        <div className="border border-slate-800 rounded-xl p-6 bg-slate-950/40 flex flex-col items-center justify-center min-h-[160px]">
          {isRecording ? (
            <div className="flex flex-col items-center gap-3">
              <div className="flex items-center gap-2 text-rose-400 font-mono text-xl font-semibold">
                <span className="w-3.5 h-3.5 rounded-full bg-rose-500 recording-pulse" />
                {formatSeconds(recordSeconds)}
              </div>
              <p className="text-xs text-slate-400">Speak into microphone with natural pauses...</p>
              <button
                onClick={stopRecording}
                className="mt-1 px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-sm font-medium flex items-center gap-2 shadow-lg transition-all"
              >
                <Square className="w-4 h-4" /> Stop Recording
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <button
                onClick={startRecording}
                disabled={isLoading}
                className="p-3.5 bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-400 border border-indigo-500/30 hover:border-indigo-500/50 rounded-full transition-all group"
              >
                <Mic className="w-6 h-6 group-hover:scale-110 transition-transform" />
              </button>
              <p className="text-sm font-medium text-slate-200">Record from Microphone</p>
              <p className="text-xs text-slate-500">Live in-browser audio capture</p>
            </div>
          )}
        </div>
      </div>

      {/* Selected Audio Preview Bar */}
      {selectedFile && (
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <button
              onClick={togglePreview}
              className="p-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
            >
              {isPlayingPreview ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
            <div>
              <p className="text-sm font-medium text-white truncate max-w-xs sm:max-w-md">
                {selectedFile.name}
              </p>
              <p className="text-xs text-slate-400">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • {selectedFile.type || 'audio/wav'}
              </p>
            </div>
            {audioUrl && (
              <audio
                ref={audioPreviewRef}
                src={audioUrl}
                onEnded={() => setIsPlayingPreview(false)}
                className="hidden"
              />
            )}
          </div>

          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-medium text-sm rounded-xl shadow-lg shadow-indigo-600/20 flex items-center gap-2 transition-all disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Processing Audio...
              </>
            ) : (
              <>
                Clean Pauses & Stammers
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};
