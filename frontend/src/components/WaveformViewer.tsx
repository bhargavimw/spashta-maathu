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
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;

    const wsRegions = RegionsPlugin.create();
    regionsPluginRef.current = wsRegions;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor,
      progressColor,
      cursorColor: '#f43f5e',
      cursorWidth: 2,
      height: 80,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      minPxPerSec: zoomLevel,
      plugins: [wsRegions],
    });

    ws.load(audioUrl);

    ws.on('ready', () => {
      setLoaded(true);
      setDuration(ws.getDuration());

      if (removedRegions.length > 0) {
        wsRegions.clearRegions();
        removedRegions.forEach((reg, index) => {
          wsRegions.addRegion({
            id: `cut-${index}`,
            start: reg.start,
            end: reg.end,
            color: 'rgba(239, 68, 68, 0.45)',
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
    if (syncTime !== undefined && wavesurferRef.current && loaded) {
      const current = wavesurferRef.current.getCurrentTime();
      if (Math.abs(current - syncTime) > 0.1) {
        const dur = wavesurferRef.current.getDuration();
        if (dur > 0) {
          wavesurferRef.current.seekTo(Math.min(1.0, syncTime / dur));
        }
      }
    }
  }, [syncTime, loaded]);

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
    <div className="glass p-4">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-200">{title}</span>
          {removedRegions.length > 0 && (
            <span className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20 text-xs flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
              {removedRegions.length} cuts marked
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
          <span>{formatTime(currentTime)}</span>
          <span className="mx-1 text-slate-600">/</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      <div ref={containerRef} className="rounded-lg overflow-hidden bg-slate-950/60 p-1 mb-3 border border-slate-800/50" />

      <div className="flex items-center justify-between pt-1 text-slate-300">
        <div className="flex items-center gap-2">
          <button
            onClick={togglePlay}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title={isPlaying ? 'Pause' : 'Play'}
            disabled={!loaded}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button
            onClick={handleRestart}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Restart"
            disabled={!loaded}
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={() => handleZoom(-15)}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Zoom out"
          >
            <ZoomOut className="w-3 h-3" />
          </button>
          <span className="text-xs font-mono text-slate-500 px-1">{zoomLevel}px/s</span>
          <button
            onClick={() => handleZoom(15)}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Zoom in"
          >
            <ZoomIn className="w-3 h-3" />
          </button>
        </div>
      </div>
    </div>
  );
};
