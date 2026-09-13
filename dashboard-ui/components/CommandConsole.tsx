'use client';

import React, { useState } from 'react';
import { Mic, Send, AlertOctagon, Terminal, Sparkles, CheckCircle2 } from 'lucide-react';

interface CommandConsoleProps {
  transcript: string;
  settledText: string;
  plan: string[];
  planIndex: number;
  onSendInstruction: (instruction: string) => void;
  onInterrupt: () => void;
}

const QUICK_PROMPTS = [
  'Set the dinner table',
  'Retrieve cutlery from drawer',
  'Hook and drag plate to mat',
  'Halt and return to home pose',
];

export const CommandConsole: React.FC<CommandConsoleProps> = ({
  transcript,
  settledText,
  plan,
  planIndex,
  onSendInstruction,
  onInterrupt,
}) => {
  const [inputText, setInputText] = useState('');

  const handleSend = () => {
    if (inputText.trim()) {
      onSendInstruction(inputText.trim());
      setInputText('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  return (
    <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 flex flex-col gap-4 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <h2 className="text-sm font-bold text-white tracking-wide">COMMAND CONSOLE & VLM PLAN</h2>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400">
          Speechmatics ASR • τ=0.774s
        </span>
      </div>

      {/* Plan Step Execution List */}
      <div className="flex flex-col gap-1.5 font-mono text-xs">
        <span className="text-zinc-500 text-[10px] uppercase font-bold tracking-wider">Active Execution Plan</span>
        <div className="max-h-36 overflow-y-auto space-y-1 pr-1">
          {plan && plan.length > 0 ? (
            plan.map((step, idx) => {
              const isCompleted = idx < planIndex;
              const isActive = idx === planIndex;

              return (
                <div
                  key={idx}
                  className={`px-3 py-1.5 rounded flex items-center justify-between border transition-all ${
                    isActive
                      ? 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300 font-bold shadow-[0_0_10px_rgba(16,185,129,0.15)]'
                      : isCompleted
                      ? 'bg-zinc-900/40 border-zinc-800/60 text-zinc-500'
                      : 'bg-zinc-900/20 border-zinc-800/40 text-zinc-600'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-zinc-500">{(idx + 1).toString().padStart(2, '0')}.</span>
                    <span className="truncate">{step}</span>
                  </div>
                  {isCompleted ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                  ) : isActive ? (
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                  ) : null}
                </div>
              );
            })
          ) : (
            <div className="text-zinc-600 italic py-2">No active plan tokens</div>
          )}
        </div>
      </div>

      {/* Voice Transcript Display */}
      <div className="bg-black/90 border border-zinc-800/90 rounded-lg p-3 flex flex-col gap-1.5 font-mono text-xs">
        <div className="flex items-center justify-between text-[10px] text-zinc-500">
          <span className="flex items-center gap-1 text-emerald-400">
            <Mic className="w-3 h-3 text-emerald-400" /> Speechmatics Settling Buffer
          </span>
          <span>Settled (τ=0.774s)</span>
        </div>
        <div className="min-h-7 leading-relaxed">
          <span className="text-emerald-400 font-semibold">{settledText || transcript || 'Ready for instruction...'}</span>
          <span className="text-zinc-500 ml-1">
            {transcript && settledText && transcript.replace(settledText, '')}
          </span>
        </div>
      </div>

      {/* Quick Prompt Chips */}
      <div className="flex flex-wrap gap-1.5">
        {QUICK_PROMPTS.map((prompt, idx) => (
          <button
            key={idx}
            onClick={() => onSendInstruction(prompt)}
            className="text-[11px] font-mono px-2.5 py-1 rounded-md bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 text-zinc-300 transition-all"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Input & Action Buttons */}
      <div className="flex items-center gap-2">
        <div className="relative flex-grow">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type task instruction..."
            className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-xs font-mono text-white placeholder-zinc-500 focus:outline-none focus:border-emerald-500/80 transition-all"
          />
        </div>

        <button
          onClick={handleSend}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-mono font-bold flex items-center gap-1.5 transition-all shadow-[0_0_12px_rgba(16,185,129,0.3)]"
        >
          <Send className="w-3.5 h-3.5" />
          Send
        </button>

        <button
          onClick={onInterrupt}
          className="px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-mono font-bold flex items-center gap-1.5 transition-all shadow-[0_0_15px_rgba(244,67,54,0.3)] animate-pulse"
        >
          <AlertOctagon className="w-3.5 h-3.5" />
          INTERRUPT
        </button>
      </div>
    </div>
  );
};