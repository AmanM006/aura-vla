"""
scripts/pipeline_collector_trainer.py — Master Background Pipeline Orchestrator.

Automated 2-stage execution:
  Stage 1: Resumes and collects the remaining demonstration episodes (seeds 3051..3299)
           up to 300 total episodes, checkpointing every 25 episodes to diffusion_300_dataset.npz.
  Stage 2: Automatically triggers train_diffusion() on the full 300-episode dataset,
           exports to OpenVINO IR, and benchmarks heterogeneous inference on CPU & iGPU.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np

# Defensive UTF-8 terminal handling
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Disable osmesa on Windows
if sys.platform == "win32" and os.environ.get("MUJOCO_GL") in ("osmesa", "OSMESA"):
    del os.environ["MUJOCO_GL"]

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.controller import ScriptedController
from envs.randomize import DomainRandomizer
from envs.task import TaskMonitor
from scripts.collect_demos import record_episode, save_hdf5
from policy.train_diffusion import train_diffusion, export_and_benchmark_openvino

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "evidence" / "pipeline_300_progress.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Pipeline300")


def main():
    target_total = 300
    seed_base = 3000
    demos_dir = ROOT / "data" / "demos"
    demos_dir.mkdir(parents=True, exist_ok=True)
    npz_path = demos_dir / "diffusion_300_dataset.npz"
    evidence_dir = ROOT / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    logger.info("═" * 65)
    logger.info("  AURA-VLA: 300-Episode Data Collection & Auto-Retraining Pipeline")
    logger.info("═" * 65)

    # 1. Scan existing episodes
    existing_h5 = sorted(list(demos_dir.glob("episode_*.hdf5")))
    existing_count = len(existing_h5)
    logger.info("Found %d existing demonstration episodes in %s", existing_count, demos_dir)

    all_episodes_meta = []
    # Load metadata of existing episodes for NPZ packaging
    for h5_file in existing_h5:
        try:
            with h5py.File(h5_file, "r") as f:
                all_episodes_meta.append({
                    "obs_state": f["data"]["observation.state"][:],
                    "action": f["data"]["action"][:],
                    "instruction": str(f.attrs.get("instruction", "set the dinner table")),
                    "num_frames": int(f.attrs.get("num_frames", len(f["data"]["observation.state"]))),
                })
        except Exception as e:
            logger.warning("Error reading %s: %s", h5_file, e)

    # 2. Stage 1: Data Collection
    xml_path = ROOT / "scene" / "dinner_table.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    renderers = {
        "overhead": mujoco.Renderer(model, height=128, width=128),
        "left_wrist": mujoco.Renderer(model, height=128, width=128),
        "right_wrist": mujoco.Renderer(model, height=128, width=128),
    }
    cam_ids = {
        "overhead": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam"),
        "left_wrist": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "left_wrist_cam"),
        "right_wrist": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "right_wrist_cam"),
    }

    collected = existing_count
    needed = max(0, target_total - collected)
    logger.info("Target: %d episodes total (%d remaining to collect).", target_total, needed)

    t0 = time.time()
    try:
        for i in range(needed):
            ep_idx = collected
            seed = seed_base + ep_idx
            ep = record_episode(model, data, seed=seed, renderers=renderers, cam_ids=cam_ids)

            if ep is not None:
                h5_file = demos_dir / f"episode_{ep_idx:05d}.hdf5"
                save_hdf5(ep, h5_file)
                all_episodes_meta.append({
                    "obs_state": ep["obs_state"],
                    "action": ep["action"],
                    "instruction": ep["instruction"],
                    "num_frames": ep["num_frames"],
                })
                collected += 1

                elapsed = time.time() - t0
                rate = (i + 1) / max(0.1, elapsed)
                eta_m = (needed - (i + 1)) / max(0.001, rate) / 60.0
                logger.info(
                    "[EP %3d/%3d] Seed %5d | %.2f ep/s | ETA: %.1fm | Saved %s",
                    collected,
                    target_total,
                    seed,
                    rate,
                    eta_m,
                    h5_file.name,
                )
            else:
                logger.warning("[SKIP] Seed %d did not achieve quality threshold", seed)

            # Checkpoint NPZ every 25 episodes
            if collected % 25 == 0 or collected == target_total:
                np.savez_compressed(
                    str(npz_path),
                    obs_states=np.concatenate([e["obs_state"] for e in all_episodes_meta], axis=0),
                    actions=np.concatenate([e["action"] for e in all_episodes_meta], axis=0),
                    instructions=np.array([e["instruction"] for e in all_episodes_meta]),
                    episode_lengths=np.array([e["num_frames"] for e in all_episodes_meta], dtype=np.int32),
                )
                logger.info("💾 Dataset Checkpoint saved -> %s (%d episodes)", npz_path, collected)

    finally:
        for r in renderers.values():
            r.close()

    total_time_col = time.time() - t0
    logger.info(
        "Stage 1 Complete: %d episodes now available in %s (Collection run: %.1f mins)",
        collected,
        demos_dir,
        total_time_col / 60.0,
    )

    # 3. Stage 2: Retrain Diffusion Policy on full dataset
    logger.info("═" * 65)
    logger.info("  STAGE 2: Retraining Continuous VLA Diffusion Policy (300 Episodes)...")
    logger.info("═" * 65)

    out_checkpoint = ROOT / "models" / "diffusion_policy.pt"
    ir_dir = ROOT / "models" / "diffusion_openvino"
    ir_dir.mkdir(parents=True, exist_ok=True)

    train_meta = train_diffusion(
        demos_dir=demos_dir,
        out_checkpoint=out_checkpoint,
        epochs=15,
        batch_size=32,
        lr=1e-4,
    )

    # 4. Stage 3: Export OpenVINO IR & Benchmark
    logger.info("═" * 65)
    logger.info("  STAGE 3: OpenVINO Heterogeneous IR Export & Benchmarking...")
    logger.info("═" * 65)

    bench_meta = export_and_benchmark_openvino(
        checkpoint_path=out_checkpoint,
        ir_out_dir=ir_dir,
    )

    final_payload = {
        "status": "COMPLETED",
        "total_episodes": collected,
        "dataset_npz": str(npz_path),
        "collection_time_s": round(total_time_col, 2),
        "training": train_meta,
        "openvino_benchmark": bench_meta,
        "checkpoint_path": str(out_checkpoint),
        "ir_directory": str(ir_dir),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    complete_file = evidence_dir / "pipeline_300_complete.json"
    complete_file.write_text(json.dumps(final_payload, indent=2))
    logger.info("🎉 Master Pipeline Finished Successfully! Final evidence saved to %s", complete_file)


if __name__ == "__main__":
    main()