'use client';

import React from 'react';
import { Cpu, Wifi, WifiOff, Activity } from 'lucide-react';

interface HeaderProps {
  connected: boolean;
  phase: string;
  uptime: number;
  isOfflineReplay?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ connected, phase, uptime, isOfflineReplay }) => {
  const formatUptime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <header className="border-b border-zinc-800/80 bg-black/90 backdrop-blur-md px-6 py-3 sticky top-0 z-50 flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-emerald-500/20 to-cyan-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-mono font-bold text-lg shadow-[0_0_15px_rgba(16,185,129,0.2)]">
          λ
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-bold tracking-tight text-white text-base">AURA-VLA</h1>
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-500/30 font-semibold">
              Dual SO-101 Robotics
            </span>
          </div>
          <p className="text-xs text-zinc-400 font-mono">
            Intel® Core™ Ultra Heterogeneous Inference • OpenVINO™ INT8
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 bg-zinc-950/80 border border-zinc-800/80 px-3 py-1.5 rounded-lg text-xs font-mono">
        <span className="text-zinc-500">PHASE:</span>
        <span className="text-emerald-400 font-semibold animate-pulse">{phase || 'Initializing...'}</span>
        <span className="text-zinc-700">|</span>
        <span className="text-zinc-500">SIM:</span>
        <span className="text-zinc-300">MuJoCo 500Hz</span>
        <span className="text-zinc-700">|</span>
        <span className="text-zinc-500">UPTIME:</span>
        <span className="text-zinc-300">{formatUptime(uptime)}</span>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden md:flex items-center gap-2 bg-blue-950/40 border border-blue-800/40 px-2.5 py-1 rounded-md text-xs text-blue-400 font-mono">
          <Cpu className="w-3.5 h-3.5 text-blue-400" />
          <span>Core Ultra iGPU + NPU</span>
        </div>

        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono font-medium border ${
          connected 
            ? 'bg-emerald-950/60 text-emerald-400 border-emerald-500/40 shadow-[0_0_10px_rgba(16,185,129,0.2)]' 
            : isOfflineReplay
            ? 'bg-amber-950/60 text-amber-300 border-amber-500/40 shadow-[0_0_10px_rgba(245,158,11,0.2)]'
            : 'bg-emerald-950/60 text-emerald-300 border-emerald-500/30'
        }`}>
          {connected ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
              <Wifi className="w-3.5 h-3.5" />
              <span>LIVE WS</span>
            </>
          ) : isOfflineReplay ? (
            <>
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              <Activity className="w-3.5 h-3.5" />
              <span>EVALUATION RUN (REPLAY)</span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <Activity className="w-3.5 h-3.5" />
              <span>ACTIVE BENCHMARK</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
};