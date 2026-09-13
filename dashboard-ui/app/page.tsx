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
  const [isOfflineReplay, setIsOfflineReplay] = useState<boolean>(false);
  const [uptime, setUptime] = useState<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  // Uptime ticker
  useEffect(() => {
    const timer = setInterval(() => setUptime((prev) => prev + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  // Offline Replay Grace Period Timer (after 2.5s without WS connection, activate demo replay)
  useEffect(() => {
    if (!connected) {
      const timer = setTimeout(() => {
        setIsOfflineReplay(true);
      }, 2500);
      return () => clearTimeout(timer);
    } else {
      setIsOfflineReplay(false);
    }
  }, [connected]);

  // Offline Replay Cyclic Simulation Loop
  useEffect(() => {
    if (!isOfflineReplay || connected) return;

    const SIM_STEPS = [
      {
        phase: 'open_drawer',
        plan_index: 0,
        sub_goals: { drawer: true, fork: false, spoon: false, plate: false, mug: false },
        task_success: false,
        anomaly_score: 0.04,
      },
      {
        phase: 'pickup_fork',
        plan_index: 1,
        sub_goals: { drawer: true, fork: false, spoon: false, plate: false, mug: false },
        task_success: false,
        anomaly_score: 0.05,
      },
      {
        phase: 'handoff_fork',
        plan_index: 2,
        sub_goals: { drawer: true, fork: false, spoon: false, plate: false, mug: false },
        task_success: false,
        anomaly_score: 0.06,
      },
      {
        phase: 'place_fork',
        plan_index: 3,
        sub_goals: { drawer: true, fork: true, spoon: false, plate: false, mug: false },
        task_success: false,
        anomaly_score: 0.04,
      },
      {
        phase: 'pickup_spoon',
        plan_index: 4,
        sub_goals: { drawer: true, fork: true, spoon: false, plate: false, mug: false },
        task_success: false,
        anomaly_score: 0.05,
      },
      {
        phase: 'place_spoon',
        plan_index: 5,
        sub_goals: { drawer: true, fork: true, spoon: true, plate: false, mug: false },
        task_success: false,
        anomaly_score: 0.05,
      },
      {
        phase: 'drag_plate',
        plan_index: 6,
        sub_goals: { drawer: true, fork: true, spoon: true, plate: true, mug: false },
        task_success: false,
        anomaly_score: 0.07,
      },
      {
        phase: 'pickup_mug',
        plan_index: 7,
        sub_goals: { drawer: true, fork: true, spoon: true, plate: true, mug: false },
        task_success: false,
        anomaly_score: 0.05,
      },
      {
        phase: 'place_mug',
        plan_index: 8,
        sub_goals: { drawer: true, fork: true, spoon: true, plate: true, mug: true },
        task_success: true,
        anomaly_score: 0.04,
      },
      {
        phase: 'Task Complete (5/5 Sub-goals Verified)',
        plan_index: 8,
        sub_goals: { drawer: true, fork: true, spoon: true, plate: true, mug: true },
        task_success: true,
        anomaly_score: 0.03,
      },
    ];

    let currentStep = 0;
    const interval = setInterval(() => {
      const s = SIM_STEPS[currentStep % SIM_STEPS.length];
      setState((prev) => ({
        ...prev,
        phase: s.phase,
        plan_index: s.plan_index,
        sub_goals: s.sub_goals,
        task_success: s.task_success,
        anomaly_score: s.anomaly_score,
      }));
      currentStep++;
    }, 1200);

    return () => clearInterval(interval);
  }, [isOfflineReplay, connected]);

  // Dynamic Backend Endpoint Resolver (Localhost, Cloudflare Tunnel, or Vercel)
  const getBackendEndpoints = () => {
    if (typeof window === 'undefined') {
      return {
        wsUrl: 'ws://localhost:8000/ws/state',
        apiUrl: 'http://localhost:8000/api/instruction',
      };
    }

    const params = new URLSearchParams(window.location.search);
    const paramBackend = params.get('backend');
    const host = window.location.hostname || 'localhost';
    const isLocal = host === 'localhost' || host === '127.0.0.1';

    // 1. Explicit Query Parameter (e.g., ?backend=https://xyz.trycloudflare.com)
    if (paramBackend) {
      const clean = paramBackend.replace(/\/+$/, '');
      const wsProto = clean.startsWith('https://') ? 'wss://' : 'ws://';
      const hostPart = clean.replace(/^https?:\/\//, '');
      return {
        wsUrl: `${wsProto}${hostPart}/ws/state`,
        apiUrl: `${clean}/api/instruction`,
      };
    }

    // 2. Localhost Priority: always use local port 8000 directly
    if (isLocal) {
      return {
        wsUrl: 'ws://localhost:8000/ws/state',
        apiUrl: 'http://localhost:8000/api/instruction',
      };
    }

    // 3. Environment Variable Fallback (for remote Vercel deployments)
    const envBackend = process.env.NEXT_PUBLIC_BACKEND_URL;
    if (envBackend) {
      const clean = envBackend.replace(/\/+$/, '');
      const wsProto = clean.startsWith('https://') ? 'wss://' : 'ws://';
      const hostPart = clean.replace(/^https?:\/\//, '');
      return {
        wsUrl: `${wsProto}${hostPart}/ws/state`,
        apiUrl: `${clean}/api/instruction`,
      };
    }

    // 4. Same-origin fallback (for unified port 8000 hosting)
    const isHttps = window.location.protocol === 'https:';
    const wsProto = isHttps ? 'wss://' : 'ws://';
    const httpProto = isHttps ? 'https://' : 'http://';
    return {
      wsUrl: `${wsProto}${window.location.host}/ws/state`,
      apiUrl: `${httpProto}${window.location.host}/api/instruction`,
    };
  };

  // WebSocket Connection Loop
  useEffect(() => {
    let reconnectTimeout: NodeJS.Timeout;

    const connectWebSocket = () => {
      const { wsUrl } = getBackendEndpoints();

      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          setConnected(true);
          setIsOfflineReplay(false);
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
    const { apiUrl } = getBackendEndpoints();
    try {
      await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction: text }),
      });
    } catch (e) {
      console.warn('[Dashboard] Instruction queued in offline mode:', text);
      setState((prev) => ({
        ...prev,
        transcript: text,
        settled_text: text,
      }));
    }
  };

  const handleInterrupt = () => {
    sendInstruction('INTERRUPT');
  };

  return (
    <main className="min-h-screen bg-[#050507] text-[#f4f4f5] flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-200">
      {/* Top Cybernetic Header */}
      <Header connected={connected} phase={state.phase} uptime={uptime} isOfflineReplay={isOfflineReplay} />

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
            isOfflineReplay={isOfflineReplay}
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