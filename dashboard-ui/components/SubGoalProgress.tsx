'use client';

import React from 'react';
import { CheckCircle2, Circle, Clock, Check } from 'lucide-react';

interface SubGoalProgressProps {
  subGoals: {
    drawer: boolean;
    fork: boolean;
    spoon: boolean;
    plate: boolean;
    mug: boolean;
  };
  taskSuccess: boolean;
  phase: string;
}

interface GoalMeta {
  key: keyof SubGoalProgressProps['subGoals'];
  title: string;
  desc: string;
  phaseTag: string;
}

const GOALS: GoalMeta[] = [
  { key: 'drawer', title: 'Open Drawer', desc: 'Slide joint travel > 70mm', phaseTag: 'open_drawer' },
  { key: 'fork', title: 'Place Fork', desc: 'Bimanual handoff & placemat left', phaseTag: 'place_fork' },
  { key: 'spoon', title: 'Place Spoon', desc: 'Direct pick from drawer & placemat right', phaseTag: 'place_spoon' },
  { key: 'plate', title: 'Drag Plate', desc: 'Hook inner rim & slide to placemat', phaseTag: 'drag_plate' },
  { key: 'mug', title: 'Place Mug', desc: 'Grasp & upright placement to plate right', phaseTag: 'place_mug' },
];

export const SubGoalProgress: React.FC<SubGoalProgressProps> = ({ subGoals, taskSuccess, phase }) => {
  const completedCount = Object.values(subGoals).filter(Boolean).length;
  const progressPct = Math.round((completedCount / GOALS.length) * 100);

  return (
    <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 flex flex-col gap-3 shadow-xl">
      {/* Header & Overall Counter */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-white tracking-wide">TASK SUB-GOAL VERIFICATION</h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400">
              5 Predicates
            </span>
          </div>
          <p className="text-xs text-zinc-500 font-mono mt-0.5">Automated TaskMonitor predicates checked @ 500Hz</p>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="text-right">
            <div className="text-base font-bold text-white leading-none">{completedCount}/5</div>
            <div className="text-[10px] text-zinc-500">{progressPct}% PASS</div>
          </div>
          <div className="w-10 h-10 rounded-full border-2 border-zinc-800 flex items-center justify-center relative overflow-hidden bg-black">
            <div
              className={`text-xs font-bold font-mono ${taskSuccess ? 'text-emerald-400' : 'text-zinc-300'}`}
            >
              {taskSuccess ? '✓' : `${progressPct}%`}
            </div>
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full h-1.5 bg-zinc-900 rounded-full overflow-hidden border border-zinc-800/80">
        <div
          className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all duration-300 shadow-[0_0_10px_rgba(16,185,129,0.5)]"
          style={{ width: `${progressPct}%` }}
        />
      </div>

      {/* Sub-Goal Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-5 gap-2 mt-1">
        {GOALS.map((g, idx) => {
          const isDone = subGoals[g.key];
          const isCurrent = !isDone && phase?.toLowerCase().includes(g.phaseTag);

          return (
            <div
              key={g.key}
              className={`relative flex flex-col justify-between p-2.5 rounded-lg border transition-all ${
                isDone
                  ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-300 shadow-[0_0_12px_rgba(16,185,129,0.1)]'
                  : isCurrent
                  ? 'bg-amber-950/20 border-amber-500/50 text-amber-300 animate-pulse'
                  : 'bg-zinc-900/50 border-zinc-800/80 text-zinc-400'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-mono font-semibold text-zinc-500">0{idx + 1}</span>
                {isDone ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : isCurrent ? (
                  <Clock className="w-4 h-4 text-amber-400 animate-spin" />
                ) : (
                  <Circle className="w-3.5 h-3.5 text-zinc-600" />
                )}
              </div>

              <div>
                <div className="text-xs font-bold leading-tight truncate">{g.title}</div>
                <div className="text-[10px] text-zinc-500 truncate mt-0.5 font-mono">{g.desc}</div>
              </div>

              <div className="mt-2 pt-1.5 border-t border-zinc-800/60 flex items-center justify-between text-[9px] font-mono">
                <span className="text-zinc-500">STATE:</span>
                <span className={`font-bold ${isDone ? 'text-emerald-400' : isCurrent ? 'text-amber-400' : 'text-zinc-600'}`}>
                  {isDone ? 'COMPLETED' : isCurrent ? 'IN PROGRESS' : 'PENDING'}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};