import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import { Play, Pause, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import { RemovedRegion } from '../types';

interface WaveformViewerProps {
  audioUrl: string;
  removedRegions?: RemovedRegion[];
  title?: string;
  waveColor?: string;
  progressColor?: string;
  onTimeUpdate?: (currentTime: number) => void;
  syncTime?: number;
}

export const WaveformViewer: React.FC<WaveformViewerProps> = ({
  audioUrl,
  removedRegions = [],
  title = 'Waveform View',
  waveColor = '#64748b',
  progressColor = '#6366f1',
  onTimeUpdate,
  syncTime,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsPluginRef = useRef<any>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [zoomLevel, setZoomLevel] = useState(20);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize Regions Plugin
    const wsRegions = RegionsPlugin.create();
    regionsPluginRef.current = wsRegions;

    // Initialize WaveSurfer
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor,
      progressColor,
      cursorColor: '#f43f5e',
      cursorWidth: 2,
      height: 90,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      minPxPerSec: zoomLevel,
      plugins: [wsRegions],
    });

    ws.load(audioUrl);

    ws.on('ready', () => {
      setDuration(ws.getDuration());

      // Add red overlay regions for removed pauses/stammers
      if (removedRegions.length > 0) {
        wsRegions.clearRegions();
        removedRegions.forEach((reg, index) => {
          wsRegions.addRegion({
            id: `cut-${index}`,
            start: reg.start,
            end: reg.end,
            color: 'rgba(239, 68, 68, 0.45)', // Red overlay
            drag: false,
            resize: false,
            content: document.createElement('div'),
          });
        });
      }
    });

    ws.on('audioprocess', () => {
      const time = ws.getCurrentTime();
      setCurrentTime(time);
      if (onTimeUpdate) onTimeUpdate(time);
    });

    ws.on('seeking', () => {
      const time = ws.getCurrentTime();
      setCurrentTime(time);
      if (onTimeUpdate) onTimeUpdate(time);
    });

    ws.on('play', () => setIsPlaying(true));
    ws.on('pause', () => setIsPlaying(false));
    ws.on('finish', () => setIsPlaying(false));

    wavesurferRef.current = ws;

    return () => {
      ws.destroy();
    };
  }, [audioUrl, removedRegions, waveColor, progressColor]);

  // Sync external seek time if provided
  useEffect(() => {
    if (syncTime !== undefined && wavesurferRef.current) {
      const current = wavesurferRef.current.getCurrentTime();
      if (Math.abs(current - syncTime) > 0.1) {
        const dur = wavesurferRef.current.getDuration();
        if (dur > 0) {
          wavesurferRef.current.seekTo(Math.min(1.0, syncTime / dur));
        }
      }
    }
  }, [syncTime]);

  const togglePlay = () => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause();
    }
  };

  const handleRestart = () => {
    if (wavesurferRef.current) {
      wavesurferRef.current.seekTo(0);
      wavesurferRef.current.play();
    }
  };

  const handleZoom = (delta: number) => {
    const nextZoom = Math.max(10, Math.min(150, zoomLevel + delta));
    setZoomLevel(nextZoom);
    if (wavesurferRef.current) {
      wavesurferRef.current.zoom(nextZoom);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`;
  };

  return (
    <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-200">{title}</span>
          {removedRegions.length > 0 && (
            <span className="text-xs px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500 inline-block" />
              {removedRegions.length} cuts marked in red
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
          <span>{formatTime(currentTime)}</span>
          <span>/</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* WaveSurfer Container */}
      <div ref={containerRef} className="rounded-lg overflow-hidden bg-slate-900/60 p-1 mb-3" />

      {/* Controls Bar */}
      <div className="flex items-center justify-between pt-1 text-slate-300">
        <div className="flex items-center gap-2">
          <button
            onClick={togglePlay}
            className="p-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button
            onClick={handleRestart}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Restart to beginning"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => handleZoom(-15)}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Zoom out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="text-xs font-mono text-slate-500 px-1">{zoomLevel}px/s</span>
          <button
            onClick={() => handleZoom(15)}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Zoom in"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
