'use client';

import React from 'react';
import { ShieldAlert, ShieldCheck, AlertTriangle, RefreshCw } from 'lucide-react';

interface AnomalibDefectRadarProps {
  score?: number;
  isDefect?: boolean;
  hotspot?: string;
  device?: string;
}

export const AnomalibDefectRadar: React.FC<AnomalibDefectRadarProps> = ({
  score = 0.08,
  isDefect = false,
  hotspot = 'nominal',
  device = 'OpenVINO (iGPU)',
}) => {
  const safeScore = Math.max(0, Math.min(1, score || 0));
  const scorePct = Math.round(safeScore * 100);
  const thresholdPct = 65;

  return (
    <div className={`border rounded-xl p-4 flex flex-col gap-3 shadow-xl transition-all ${
      isDefect 
        ? 'bg-rose-950/20 border-rose-500/50 shadow-[0_0_20px_rgba(244,67,54,0.2)]' 
        : 'bg-zinc-950 border-zinc-800'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isDefect ? (
            <ShieldAlert className="w-4 h-4 text-rose-400 animate-bounce" />
          ) : (
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          )}
          <h2 className="text-sm font-bold text-white tracking-wide">INTEL ANOMALIB DEFECT MONITOR</h2>
        </div>

        <div className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-bold border flex items-center gap-1.5 ${
          isDefect
            ? 'bg-rose-900/60 text-rose-300 border-rose-500/60 animate-pulse'
            : 'bg-emerald-950/60 text-emerald-400 border-emerald-500/40'
        }`}>
          {isDefect ? (
            <>
              <AlertTriangle className="w-3 h-3 text-rose-400" />
              <span>DEFECT DETECTED</span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span>NOMINAL</span>
            </>
          )}
        </div>
      </div>

      {/* Score Meter with Threshold */}
      <div className="flex flex-col gap-1.5 font-mono">
        <div className="flex justify-between items-center text-xs">
          <span className="text-zinc-400">ANOMALY SCORE</span>
          <span className={`text-base font-bold ${isDefect ? 'text-rose-400' : safeScore > 0.45 ? 'text-amber-400' : 'text-emerald-400'}`}>
            {safeScore.toFixed(3)}
          </span>
        </div>

        <div className="relative w-full h-3 bg-zinc-900 rounded-full overflow-hidden border border-zinc-800">
          <div
            className={`h-full transition-all duration-300 ${
              isDefect ? 'bg-rose-500' : safeScore > 0.45 ? 'bg-amber-500' : 'bg-emerald-500'
            }`}
            style={{ width: `${scorePct}%` }}
          />
          {/* Threshold marker line at 65% */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-rose-400 z-10"
            style={{ left: `${thresholdPct}%` }}
            title="Replan Trigger Threshold (0.65)"
          />
        </div>

        <div className="flex justify-between text-[10px] text-zinc-500">
          <span>0.00 (Nominal)</span>
          <span className="text-rose-400 font-semibold">0.65 REPLAN THRESHOLD</span>
          <span>1.00 (Critical)</span>
        </div>
      </div>

      {/* Metadata Row */}
      <div className="grid grid-cols-2 gap-2 pt-2 border-t border-zinc-800/80 font-mono text-xs">
        <div className="bg-zinc-900/60 p-2 rounded border border-zinc-800/60">
          <span className="text-zinc-500 text-[10px] block">DEFECT HOTSPOT</span>
          <span className="text-zinc-200 font-bold uppercase">{hotspot || 'none'}</span>
        </div>
        <div className="bg-zinc-900/60 p-2 rounded border border-zinc-800/60">
          <span className="text-zinc-500 text-[10px] block">INSPECTION ACCELERATOR</span>
          <span className="text-zinc-200 font-bold">{device}</span>
        </div>
      </div>
    </div>
  );
};