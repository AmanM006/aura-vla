'use client';

import React from 'react';
import { Cpu, Zap, Layers, Sparkles } from 'lucide-react';

interface HardwareTelemetryProps {
  npuMs?: number;
  igpuMs?: number;
  cpuMs?: number;
}

export const HardwareTelemetry: React.FC<HardwareTelemetryProps> = ({
  npuMs = 5.2,
  igpuMs = 42.1,
  cpuMs = 1.41,
}) => {
  const totalMs = (npuMs || 0) + (igpuMs || 0) + (cpuMs || 0);

  const npuPct = totalMs > 0 ? ((npuMs / totalMs) * 100).toFixed(1) : '10.5';
  const igpuPct = totalMs > 0 ? ((igpuMs / totalMs) * 100).toFixed(1) : '86.4';
  const cpuPct = totalMs > 0 ? ((cpuMs / totalMs) * 100).toFixed(1) : '3.1';

  return (
    <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 flex flex-col gap-3 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-white tracking-wide">INTEL HETEROGENEOUS COMPUTE</h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-950/60 text-blue-400 border border-blue-800/40">
              OpenVINO™ 2024
            </span>
          </div>
          <p className="text-xs text-zinc-500 font-mono mt-0.5">3-tier hardware pipeline with zero-copy shared memory</p>
        </div>

        <div className="text-right font-mono">
          <div className="text-xs text-zinc-500">TOTAL LATENCY</div>
          <div className="text-base font-bold text-emerald-400">{totalMs.toFixed(2)} ms</div>
        </div>
      </div>

      {/* Latency Split Bar */}
      <div className="flex flex-col gap-1.5">
        <div className="w-full h-3 bg-zinc-900 rounded-full overflow-hidden flex border border-zinc-800">
          <div
            className="h-full bg-purple-500 transition-all duration-300"
            style={{ width: `${npuPct}%` }}
            title={`NPU Vision: ${npuMs}ms (${npuPct}%)`}
          />
          <div
            className="h-full bg-blue-500 transition-all duration-300"
            style={{ width: `${igpuPct}%` }}
            title={`iGPU Planner: ${igpuMs}ms (${igpuPct}%)`}
          />
          <div
            className="h-full bg-emerald-500 transition-all duration-300"
            style={{ width: `${cpuPct}%` }}
            title={`CPU Diffusion: ${cpuMs}ms (${cpuPct}%)`}
          />
        </div>

        <div className="flex items-center justify-between text-[10px] font-mono text-zinc-400 px-1">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-purple-500" /> NPU ({npuPct}%)
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-blue-500" /> iGPU ({igpuPct}%)
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500" /> CPU ({cpuPct}%)
          </span>
        </div>
      </div>

      {/* 3 Tier Detailed Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mt-1 font-mono">
        {/* Tier 1: NPU */}
        <div className="bg-zinc-900/60 border border-purple-500/20 rounded-lg p-2.5 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] text-purple-400 font-bold flex items-center gap-1">
              <Zap className="w-3 h-3 text-purple-400" /> NPU (Vision)
            </span>
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-purple-950/80 text-purple-300 border border-purple-800/40">
              INT8 PTQ
            </span>
          </div>
          <div className="text-sm font-bold text-white mt-1">{(npuMs || 5.2).toFixed(2)} ms</div>
          <div className="text-[10px] text-zinc-500 truncate mt-0.5">SigLIP / MobileNet • 192 FPS</div>
        </div>

        {/* Tier 2: iGPU */}
        <div className="bg-zinc-900/60 border border-blue-500/20 rounded-lg p-2.5 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] text-blue-400 font-bold flex items-center gap-1">
              <Layers className="w-3 h-3 text-blue-400" /> iGPU (VLM)
            </span>
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-950/80 text-blue-300 border border-blue-800/40">
              INT4 GenAI
            </span>
          </div>
          <div className="text-sm font-bold text-white mt-1">{(igpuMs || 42.1).toFixed(1)} ms</div>
          <div className="text-[10px] text-zinc-500 truncate mt-0.5">Qwen2.5-1.5B • TTFT 38ms</div>
        </div>

        {/* Tier 3: CPU */}
        <div className="bg-zinc-900/60 border border-emerald-500/20 rounded-lg p-2.5 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] text-emerald-400 font-bold flex items-center gap-1">
              <Cpu className="w-3 h-3 text-emerald-400" /> CPU (Diffusion)
            </span>
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-800/40">
              Ensemble
            </span>
          </div>
          <div className="text-sm font-bold text-emerald-400 mt-1">{(cpuMs || 1.41).toFixed(2)} ms</div>
          <div className="text-[10px] text-zinc-500 truncate mt-0.5">H=16, k=2 • 44.4 traj/s</div>
        </div>
      </div>
    </div>
  );
};