'use client';

import React, { useEffect, useState, useRef } from 'react';
import { Header } from '@/components/Header';
import { CameraViewport } from '@/components/CameraViewport';
import { SubGoalProgress } from '@/components/SubGoalProgress';
import { HardwareTelemetry } from '@/components/HardwareTelemetry';
import { AnomalibDefectRadar } from '@/components/AnomalibDefectRadar';
import { CommandConsole } from '@/components/CommandConsole';
import { SimState } from '@/types/telemetry';

export default function DashboardPage() {
  const [state, setState] = useState<SimState>({
    timestamp: 0,
    phase: 'Initializing...',
    sub_goals: { drawer: false, fork: false, spoon: false, plate: false, mug: false },
    task_success: false,
    npu_ms: 5.2,
    igpu_ms: 42.1,
    cpu_ms: 1.41,
    front_cam: null,
    overhead_cam: null,
    left_wrist_cam: null,
    right_wrist_cam: null,
    transcript: 'set the table',
    settled_text: 'set the table',
    plan: [
      'open_drawer',
      'pickup_fork',
      'handoff_fork',
      'place_fork',
      'pickup_spoon',
      'place_spoon',
      'drag_plate',
      'pickup_mug',
      'place_mug',
    ],
    plan_index: 0,
    objects: {},
    anomaly_score: 0.05,
    anomaly_is_defect: false,
    anomaly_hotspot: 'nominal',
    anomaly_device: 'OpenVINO (iGPU)',
    policy_mode: 'diffusion',
    temporal_ensemble_active: true,
  });

  const [connected, setConnected] = useState<boolean>(false);
  const [uptime, setUptime] = useState<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  // Uptime ticker
  useEffect(() => {
    const timer = setInterval(() => setUptime((prev) => prev + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  // WebSocket Connection Loop
  useEffect(() => {
    let reconnectTimeout: NodeJS.Timeout;

    const connectWebSocket = () => {
      // Determine host
      const host = typeof window !== 'undefined' ? window.location.hostname || 'localhost' : 'localhost';
      const wsUrl = `ws://${host}:8000/ws/state`;

      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          setConnected(true);
          console.log('[Dashboard] WebSocket Connected to', wsUrl);
        };

        ws.onmessage = (event) => {
          try {
            const data: SimState = JSON.parse(event.data);
            setState(data);
          } catch (err) {
            console.error('[Dashboard] Error parsing state:', err);
          }
        };

        ws.onclose = () => {
          setConnected(false);
          reconnectTimeout = setTimeout(connectWebSocket, 2000);
        };

        ws.onerror = () => {
          ws.close();
        };
      } catch (e) {
        reconnectTimeout = setTimeout(connectWebSocket, 2000);
      }
    };

    connectWebSocket();

    return () => {
      clearTimeout(reconnectTimeout);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // REST API Instruction Sender
  const sendInstruction = async (text: string) => {
    const host = typeof window !== 'undefined' ? window.location.hostname || 'localhost' : 'localhost';
    try {
      await fetch(`http://${host}:8000/api/instruction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction: text }),
      });
    } catch (e) {
      console.error('Error posting instruction:', e);
    }
  };

  const handleInterrupt = () => {
    sendInstruction('INTERRUPT');
  };

  return (
    <main className="min-h-screen bg-[#050507] text-[#f4f4f5] flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-200">
      {/* Top Cybernetic Header */}
      <Header connected={connected} phase={state.phase} uptime={uptime} />

      {/* Main Responsive Grid Container */}
      <div className="flex-grow p-4 md:p-6 max-w-[1700px] w-full mx-auto grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column (7 cols): High-Def Cameras + Sub-Goals */}
        <div className="lg:col-span-7 flex flex-col gap-5">
          <CameraViewport
            frontCam={state.front_cam}
            isoCam={state.iso_cam}
            overheadCam={state.overhead_cam}
            leftWristCam={state.left_wrist_cam}
            rightWristCam={state.right_wrist_cam}
            phase={state.phase}
            objects={state.objects}
          />

          <SubGoalProgress
            subGoals={state.sub_goals}
            taskSuccess={state.task_success}
            phase={state.phase}
          />
        </div>

        {/* Right Column (5 cols): Hardware Telemetry + Anomalib Defect Monitor + Console */}
        <div className="lg:col-span-5 flex flex-col gap-5">
          <HardwareTelemetry
            npuMs={state.npu_ms}
            igpuMs={state.igpu_ms}
            cpuMs={state.cpu_ms}
          />

          <AnomalibDefectRadar
            score={state.anomaly_score}
            isDefect={state.anomaly_is_defect}
            hotspot={state.anomaly_hotspot}
            device={state.anomaly_device}
          />

          <CommandConsole
            transcript={state.transcript}
            settledText={state.settled_text}
            plan={state.plan}
            planIndex={state.plan_index}
            onSendInstruction={sendInstruction}
            onInterrupt={handleInterrupt}
          />
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-zinc-900 bg-black/80 px-6 py-3 text-center text-xs font-mono text-zinc-600 flex flex-wrap items-center justify-between gap-2">
        <span>AI Infra Summit Hackathon 2026 • Intel Online Track</span>
        <span>AURA-VLA: Autonomous Robotic Manipulation & Heterogeneous OpenVINO INT8 Inference</span>
      </footer>
    </main>
  );
}