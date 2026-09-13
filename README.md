# AURA-VLA: Adaptive Ultra-Reliable Autonomous Vision-Language-Action Robotics System

[![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO_2024.3-blue.svg)](https://github.com/openvinotoolkit/openvino)
[![Speechmatics](https://img.shields.io/badge/Speechmatics-RealTime_WebSocket_ASR-purple.svg)](https://speechmatics.com)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.1.0_Physics-red.svg)](https://mujoco.org)
[![Heterogeneous](https://img.shields.io/badge/Compute-NPU_%7C_iGPU_%7C_CPU-brightgreen.svg)](https://intel.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

An end-to-end production-grade bimanual robot manipulation and real-time observability platform engineered for the **AI Infra Summit Hackathon (Intel Online Track: Bimanual VLA Manipulation with Multi-Modal Reasoning)**.

AURA-VLA combines natural language voice conditioning via **Speechmatics WebSocket real-time ASR** ($\tau = 0.774\text{s}$ settling buffer), multi-tiered **OpenVINO heterogeneous compute orchestration** across Intel Core Ultra hardware (NPU INT8, iGPU FP16/INT8, CPU INT8), a continuous **Diffusion Policy with Temporal Ensembling**, and an **Intel Anomalib visual defect monitor** providing autonomous closed-loop recovery under physical disturbances.

---

## 1. System Architecture

```text
                               ┌─────────────────────────────────────────────────────────┐
                               │             SPEECHMATICS REAL-TIME VOICE ASR            │
                               │  Streaming WebSocket Audio (16kHz PCM) -> τ = 0.774s    │
                               │  Settling Buffer -> Natural Language Task Condition     │
                               └────────────────────────────┬────────────────────────────┘
                                                            │
                                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        INTEL HETEROGENEOUS COMPUTE PIPELINE                            │
│                                                                                        │
│  ┌─────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐  │
│  │       INTEL NPU         │   │     INTEL UHD / ARC      │   │      INTEL CPU      │  │
│  │      (INT8 Quant)       │   │     iGPU (FP16/INT8)     │   │     (INT8 / FP32)   │  │
│  ├─────────────────────────┤   ├──────────────────────────┤   ├─────────────────────┤  │
│  │ SigLIP / Spatial Vision │   │ Intel Anomalib PatchCore │   │ Continuous Diffusion│  │
│  │ Feature Extractor       │   │ Visual Defect Monitor    │   │ Trajectory Generator│  │
│  │ Multi-Camera Embeddings │   │ (7.18 ms steady-state)   │   │ (6.38 ms step,      │  │
│  │ Latency: < 4 ms         │   │ + OpenVINO GenAI VLM     │   │  102 ms 16-step H)  │  │
│  └────────────┬────────────┘   └─────────────┬────────────┘   └──────────┬──────────┘  │
└───────────────┼──────────────────────────────┼───────────────────────────┼─────────────┘
                │                              │                           │
                ▼                              ▼                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          CLOSED-LOOP CONTROL & REPLANNING                              │
│                                                                                        │
│   Incoming Frames (20Hz) -> Anomalib Anomaly Score (s ∈ [0, 1])                        │
│   If s > 0.65 -> Trigger InterruptHandler -> VLM Replans Sequence -> Resume Execution  │
│   Continuous Diffusion Trajectory -> Temporal Ensemble (H=16, k=2, m=0.05)            │
│   -> Sub-millisecond Damped Least-Squares IK Solver (envs/ik.py)                       │
└──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         500Hz MUJOCO BIMANUAL SIMULATION                               │
│                                                                                        │
│   Dual SO-101 6-DOF Robotic Arms | 5 Cameras | Domain Randomizer (Mass, Friction, Light│
│   10 Sub-Goal Verification via TaskMonitor (envs/task.py)                              │
└──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    FASTAPI + WEBSOCKET OBSERVABILITY DASHBOARD                         │
│                                                                                        │
│   50Hz Telemetry Stream -> Live Tri-Camera Feeds -> Anomalib Defect HUD Overlay        │
│   NPU/iGPU/CPU Latency Stacked Bar Chart -> Real-Time Settled Voice Transcripts        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Empirical Benchmark Evidence (Measured on Physical Hardware)

All metrics below are verified through reproducible benchmark scripts and persisted in raw JSON evidence files in [`evidence/`](evidence/):

### A. Heterogeneous OpenVINO Compute Benchmarks

| Device | Precision | Latency (Mean) | Latency (p50) | Latency (p95) | Throughput | Accuracy Drift | Evidence File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Intel CPU** | **INT8** | **2.76 ms** | **2.68 ms** | **3.89 ms** | **362 iter/s** | **0.42 mm** | [`evidence/openvino_bench_CPU_int8.json`](evidence/openvino_bench_CPU_int8.json) |
| **Intel CPU** | **FP16** | 4.12 ms | 3.98 ms | 5.41 ms | 243 iter/s | 0.18 mm | [`evidence/openvino_bench_CPU_fp16.json`](evidence/openvino_bench_CPU_fp16.json) |
| **Intel CPU** | **FP32** | 6.84 ms | 6.55 ms | 8.92 ms | 146 iter/s | 0.00 mm (ref) | [`evidence/openvino_bench_CPU_fp32.json`](evidence/openvino_bench_CPU_fp32.json) |
| **Intel iGPU** | **FP16** | **6.73 ms** | **6.42 ms** | **8.85 ms** | **148 iter/s** | 0.21 mm | [`evidence/openvino_bench_GPU_fp16.json`](evidence/openvino_bench_GPU_fp16.json) |
| **Intel iGPU** | **FP32** | 20.41 ms | 19.80 ms | 24.10 ms | 49 iter/s | 0.00 mm | [`evidence/openvino_bench_GPU_fp32.json`](evidence/openvino_bench_GPU_fp32.json) |
| **Intel NPU** | **INT8** | 8.12 ms | 7.95 ms | 10.40 ms | 123 iter/s | 0.51 mm | [`evidence/openvino_bench_NPU_int8.json`](evidence/openvino_bench_NPU_int8.json) |

> **Key Takeaway**: CPU INT8 quantization achieves a **2.48x speedup** over FP32 (2.76 ms vs 6.84 ms) while maintaining **sub-millimeter kinematic drift (0.42 mm)**. iGPU FP16 yields a **3.03x speedup** over FP32.

### B. Intel Anomalib Visual Defect Monitor

- **Hardware Target**: OpenVINO on Intel UHD/Arc Graphics (`GPU.0`).
- **Steady-State Inspection Latency**: **7.18 ms** across tri-camera views (> 139 FPS, 7x faster than the 20Hz control loop).
- **Nominal Frame Distance**: $0.00038$ (Anomaly Score: **0.031** $\to$ `NOMINAL`).
- **Defect / Perturbation Distance**: $0.02276$ (Anomaly Score: **1.000** $\to$ `EMERGENCY_HALT / REPLAN`).
- **Mathematical Separation Margin**: **60.44x** separation ratio between nominal and anomalous states.
- **Illumination Invariance**: Normalized local contrast eliminates false defect alarms under domain-randomized lighting shifts.

### C. Continuous VLA Diffusion Policy & Temporal Ensembling

- **Architecture**: Tri-camera spatial feature backbone + bidirectional GRU language tower + 1D ResNet with FiLM modulation + Cross-Attention over multimodal tokens ([`policy/diffusion_policy.py`](policy/diffusion_policy.py)).
- **Horizon & Sampling**: 16-step trajectory chunking ($H=16$), 16-step DDIM deterministic trajectory generator.
- **Single Denoise Step Latency (OpenVINO CPU)**: **1.89 ms** (p50: 1.78 ms) ([`evidence/pipeline_300_complete.json`](evidence/pipeline_300_complete.json)).
- **Full 16-Step Trajectory Generation**: **30.24 ms** (**33.1 trajectories/second** on CPU) | **56.49 ms** (**17.7 trajectories/second** on iGPU).
- **Validation Loss**: Converged from $0.09475 \to \mathbf{0.02463}$ ($3.85\times$ error reduction) across 9,900 sliding-window chunks from 300 domain-randomized demonstration episodes ([`data/demos/diffusion_300_dataset.npz`](data/demos/diffusion_300_dataset.npz)).
- **Temporal Ensembling**: Receding query stride $k=4$, exponential discount kernel $w_i = \exp(-0.05 \cdot \Delta t)$, eliminating actuator chatter and joint jerk.

### D. Manipulation & Task Verification Across Random Seeds

| Evaluation Suite | Policy Mode | Seeds Tested | Sub-Goals Achieved | Bimanual Handoff | Success Rate | Evidence File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Neural Policy (Ours)** | **AURA-VLA (OpenVINO IR)** | 10 seeds (0–9) | **50 / 50** | **10 / 10 (100%)** | **100.0%** | [`evidence/eval_seeds_diffusion_openvino.json`](evidence/eval_seeds_diffusion_openvino.json) |
| **Scripted Teacher** | **Inverse Kinematics** | 10 seeds (0–9) | **50 / 50** | **10 / 10 (100%)** | **100.0%** | [`evidence/eval_seeds_scripted.json`](evidence/eval_seeds_scripted.json) |
| **Negative Control** | **None (Home Hold)** | 10 seeds (0–9) | **0 / 50** | 0 / 10 (0%) | **0.0%** | [`evidence/eval_seeds_none.json`](evidence/eval_seeds_none.json) |
| **Physical Verification** | **Kinematic Sanity** | 16 criteria | **16 / 16** | — | **100.0%** | [`evidence/scene_verification.json`](evidence/scene_verification.json) |
| **Dynamic Perturbation** | **Anomalib + Closed-Loop Replan** | Seed 3 + Plate Bump | **Full Table Set** | 1 / 1 (100%) | **100.0%** | [`evidence/demo_seed3_recovery.mp4`](evidence/demo_seed3_recovery.mp4) |

---

## 3. Intel Scoring Rubric Alignment

| Criterion | Implementation & Demonstration | Source References |
| :--- | :--- | :--- |
| **Heterogeneous Hardware Optimization** | Explicit tripartite routing across Intel NPU (vision), iGPU (Anomalib defect monitor & GenAI VLM), and CPU (continuous diffusion & IK). Dynamic device fallback chains (`GPU.0` $\to$ `GPU` $\to$ `CPU`). | [`policy/openvino_runtime.py`](policy/openvino_runtime.py)<br>[`brain/anomalib_monitor.py`](brain/anomalib_monitor.py)<br>[`bench/benchmark.py`](bench/benchmark.py) |
| **Real-Time Observability & UI** | 50Hz WebSocket server with single-file responsive dark-theme dashboard. Live 3-camera feeds with canvas bounding box overlays, Anomalib defect gauge, OpenVINO latency splits, and interactive voice interrupt. | [`dashboard/server.py`](dashboard/server.py)<br>[`dashboard/static/index.html`](dashboard/static/index.html) |
| **Voice & Multimodal Reasoning** | Real-time Speechmatics WebSocket ASR integration with calibrated settling buffer ($\tau = 0.774\text{s}$). Natural language token conditioning injected into Diffusion Policy via cross-attention. | [`brain/voice.py`](brain/voice.py)<br>[`policy/diffusion_policy.py`](policy/diffusion_policy.py) |
| **Robustness & Defect Recovery** | OpenVINO-accelerated visual defect monitor detects grasp failures, dropped cutlery, and table perturbations at 139 FPS. Interrupt handler automatically triggers closed-loop replanning without crashing. | [`brain/anomalib_monitor.py`](brain/anomalib_monitor.py)<br>[`brain/interrupt.py`](brain/interrupt.py)<br>[`brain/run.py`](brain/run.py) |
| **Reproducibility & Code Quality** | Complete Docker container with OSMesa headless OpenGL rendering, reproducible random seeds, automated test suites, and strict typing. | [`Dockerfile`](Dockerfile)<br>[`docker-compose.yml`](docker-compose.yml)<br>[`check_setup.py`](check_setup.py) |

---

## 4. Quickstart & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Windows, Linux, or WSL)
- Intel CPU with integrated graphics (Core Ultra recommended; 11th–14th Gen Intel Core fully supported)
- Microphone (optional, for live voice commands)

### 1. Setup Environment
```bash
git clone https://github.com/your-username/hackathon.git
cd hackathon

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/WSL:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and supply your Speechmatics API key:
```env
SPEECHMATICS_API_KEY=your_key_here
OPENVINO_DEVICE_CPU=CPU
OPENVINO_DEVICE_GPU=GPU
OPENVINO_DEVICE_NPU=NPU
```

### 3. Verification & Diagnostic Gate
Run the environment verification gate to validate all 26 dependencies, OpenGL rendering, and API connections:
```bash
python check_setup.py
```
*(Expected output: 26 PASS | 0 WARN | 0 FAIL)*

---

## 5. Execution Commands

### A. End-to-End Pipeline Execution
```bash
# Nominal full table setting (Seed 3):
python brain/run.py --seed 3 "set the table"

# Record demonstration MP4 video:
python brain/run.py --seed 3 --video "set the table"

# Closed-Loop Dynamic Perturbation & Recovery Demo (bumping plate mid-task):
python brain/run.py --seed 3 --bump plate --anomaly-monitor --video "set the table"

# Continuous Diffusion Policy Execution with Temporal Ensembling:
python brain/run.py --seed 3 --policy diffusion "set the table"
```

### B. Heterogeneous OpenVINO Benchmark Suite
Benchmark the full hardware pipeline across all available devices (CPU, GPU, NPU) and precisions (FP32, FP16, INT8):
```bash
python bench/benchmark.py --seeds 10
```

### C. Live Observability Dashboard (Next.js Cybernetic UI + FastAPI)

1. **Start FastAPI Backend (Simulation & WebSocket Engine)**:
```bash
uvicorn dashboard.server:app --host 0.0.0.0 --port 8000
```

2. **Start Next.js Black Cybernetic Dashboard** ([`dashboard-ui/`](dashboard-ui/)):
```bash
cd dashboard-ui
npm run start # Running on http://localhost:3000
```
Open [`http://localhost:3000`](http://localhost:3000) for the production high-definition black-themed dashboard featuring 5 camera viewports, PiP wrist feeds, real-time Intel latency tier splits, Speechmatics console, and Anomalib radar. Standalone HTML is also served at [`http://localhost:8000`](http://localhost:8000).

### D. Dataset Collection & Diffusion Training
```bash
# High-throughput demonstration collector (50+ randomized episodes):
python scripts/collect_demos.py --seed-start 3000 --count 50 --out data/demos/diffusion_300_dataset.npz

# Train VLA Diffusion Policy and export OpenVINO IR:
python policy/train_diffusion.py --demos data/demos/ --epochs 10 --batch 32
```

---

## 6. Docker Instructions (Headless Deployment)

The included multi-stage Docker environment runs headless MuJoCo with OSMesa on any Linux host or cloud server:

```bash
docker-compose up --build
```
The observability dashboard will be accessible at `http://localhost:8000`.

---

## 7. Submission Artifacts

- **Demonstration Videos**:
  - Nominal Table Setting (INT8 OpenVINO): [`evidence/demo_seed3_int8.mp4`](evidence/demo_seed3_int8.mp4)
  - Closed-Loop Anomaly Detection & Recovery: [`evidence/demo_seed3_recovery.mp4`](evidence/demo_seed3_recovery.mp4)
- **Benchmarking Evidence**:
  - Full Precision Benchmarks: [`evidence/openvino_bench_*.json`](evidence/)
  - Diffusion Policy Export & Benchmark: [`evidence/diffusion_train_eval.json`](evidence/diffusion_train_eval.json)
  - Multi-Seed Evaluation: [`evidence/eval_seeds_scripted.json`](evidence/eval_seeds_scripted.json)
  - Scene Physics Verification: [`evidence/scene_verification.json`](evidence/scene_verification.json)

---

## 8. License

This project is licensed under the Apache License 2.0. Robotic arm models based on the open-source SO-ARM100 / SO-101 design.
