"""
scripts/collect_demos.py — High-throughput LeRobot HDF5 and NPZ demonstration collector
for continuous Diffusion Policy training and VLA evaluation.

Features:
  - Reusable offscreen Renderer instances across episodes (3-4x faster throughput)
  - Multi-camera recording: overhead, left wrist, right wrist (128x128x3 at 20Hz)
  - Full 12-DOF action and state trajectories
  - Natural language instruction conditioning strings per episode
  - Exports individual HDF5 episodes and aggregated NPZ dataset (data/demos/diffusion_300_dataset.npz)

Usage:
    python scripts/collect_demos.py --seed-start 3000 --count 300 --out data/demos/diffusion_300_dataset.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Defensive platform guard: osmesa is Linux/Docker only
if sys.platform == "win32" and "MUJOCO_GL" in sys.modules or True:
    import os
    if sys.platform == "win32" and os.environ.get("MUJOCO_GL") in ("osmesa", "OSMESA"):
        del os.environ["MUJOCO_GL"]

from envs.controller import ScriptedController
from envs.randomize import DomainRandomizer
from envs.task import TaskMonitor

INSTRUCTIONS = [
    "set the dinner table",
    "open drawer, pick up the fork and spoon, lay them on either side, put plate on mat and mug on the right",
    "arrange the fork, spoon, plate and mug for dinner",
    "lay out the tableware with fork on the left and spoon on the right",
    "open the top drawer and set the table completely",
]


def record_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seed: int,
    renderers: dict[str, mujoco.Renderer],
    cam_ids: dict[str, int],
    record_hz: int = 20,
) -> dict | None:
    """Runs expert controller on randomized seed and returns episode trajectory dictionary."""
    # Randomize scene
    randomizer = DomainRandomizer(seed=seed)
    rand_log = randomizer.randomize(model, data)

    # Reset controller & monitor
    ctrl = ScriptedController(model, data, rand_log)
    monitor = TaskMonitor(model, data)

    obs_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    imgs_overhead: list[np.ndarray] = []
    imgs_left: list[np.ndarray] = []
    imgs_right: list[np.ndarray] = []
    timestamps: list[float] = []

    steps_per_record = int(round(1.0 / (record_hz * model.opt.timestep)))  # 500Hz / 20Hz = 25 steps
    step_idx = 0
    max_steps = 2200

    while step_idx < max_steps:
        # Record at 20Hz
        if step_idx % steps_per_record == 0:
            obs_state = data.qpos[-12:].copy()  # 12 arm joint positions
            action = data.ctrl[:12].copy()       # 12 joint targets

            # Render 3 cameras using persistent renderer buffers
            renderers["overhead"].update_scene(data, camera=cam_ids["overhead"])
            renderers["left_wrist"].update_scene(data, camera=cam_ids["left_wrist"])
            renderers["right_wrist"].update_scene(data, camera=cam_ids["right_wrist"])

            obs_states.append(obs_state)
            actions.append(action)
            imgs_overhead.append(renderers["overhead"].render().copy())
            imgs_left.append(renderers["left_wrist"].render().copy())
            imgs_right.append(renderers["right_wrist"].render().copy())
            timestamps.append(float(data.time))

        done = ctrl.step(model, data)
        monitor.step(model, data)
        step_idx += 1

        if done:
            break

    summary = monitor.get_summary()
    success = summary.get("task_success", False)

    # Quality gate: require 5/5 or 4/5 sub-goals
    if not success and summary.get("sequencing", 0) < 4:
        return None

    instruction = INSTRUCTIONS[seed % len(INSTRUCTIONS)]

    return {
        "seed": seed,
        "instruction": instruction,
        "task_success": success,
        "sub_goals": summary.get("sub_goals", {}),
        "num_frames": len(obs_states),
        "obs_state": np.array(obs_states, dtype=np.float32),
        "action": np.array(actions, dtype=np.float32),
        "img_overhead": np.array(imgs_overhead, dtype=np.uint8),
        "img_left_wrist": np.array(imgs_left, dtype=np.uint8),
        "img_right_wrist": np.array(imgs_right, dtype=np.uint8),
        "timestamp": np.array(timestamps, dtype=np.float32),
    }


def save_hdf5(ep: dict, path: Path) -> None:
    """Save an episode dictionary to LeRobot HDF5 format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["seed"] = ep["seed"]
        f.attrs["instruction"] = ep["instruction"]
        f.attrs["task_success"] = ep["task_success"]
        f.attrs["sub_goals"] = json.dumps(ep["sub_goals"])
        f.attrs["num_frames"] = ep["num_frames"]

        data_grp = f.create_group("data")
        data_grp.create_dataset("observation.state", data=ep["obs_state"])
        data_grp.create_dataset("action", data=ep["action"])
        data_grp.create_dataset("observation.images.overhead", data=ep["img_overhead"])
        data_grp.create_dataset("observation.images.left_wrist", data=ep["img_left_wrist"])
        data_grp.create_dataset("observation.images.right_wrist", data=ep["img_right_wrist"])
        data_grp.create_dataset("timestamp", data=ep["timestamp"])


def main():
    parser = argparse.ArgumentParser(description="High-throughput LeRobot demonstration collector.")
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--out", type=str, default="data/demos/diffusion_300_dataset.npz")
    parser.add_argument("--save-hdf5", action="store_true", default=True, help="Also save individual HDF5 files")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_dir = out_path.parent if out_path.suffix else out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_path = ROOT / "scene" / "dinner_table.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Initialize reusable renderers ONCE across all 300 episodes
    print(f"\n{'═'*60}")
    print(f"  AI Infra Summit Hackathon — High-Throughput Dataset Collector")
    print(f"  Target: {args.count} episodes | Seeds: {args.seed_start}..{args.seed_start + args.count - 1}")
    print(f"  Output: {out_path}")
    print(f"{'═'*60}\n")

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

    t0 = time.time()
    collected = 0
    all_episodes = []

    try:
        for i in range(args.count):
            seed = args.seed_start + i
            ep = record_episode(model, data, seed=seed, renderers=renderers, cam_ids=cam_ids)

            if ep is not None:
                collected += 1
                if args.save_hdf5:
                    h5_file = out_dir / f"episode_{collected-1:05d}.hdf5"
                    save_hdf5(ep, h5_file)

                # Keep compact references for aggregated package
                all_episodes.append(ep)

                rate = collected / max(0.1, time.time() - t0)
                eta_s = (args.count - collected) / max(0.01, rate)
                print(f"  [EP {collected:3d}/{args.count}] Seed {seed:5d} | {rate:4.2f} ep/s | ETA: {eta_s/60:4.1f}m -> '{ep['instruction'][:35]}...'")
            else:
                print(f"  [SKIP] Seed {seed:5d} below quality threshold")

            # Periodically checkpoint the aggregated NPZ dataset every 50 episodes
            if collected > 0 and (collected % 50 == 0 or collected == args.count):
                npz_save_path = out_path if out_path.suffix == ".npz" else (out_dir / "diffusion_300_dataset.npz")
                np.savez_compressed(
                    str(npz_save_path),
                    obs_states=np.concatenate([e["obs_state"] for e in all_episodes], axis=0),
                    actions=np.concatenate([e["action"] for e in all_episodes], axis=0),
                    instructions=np.array([e["instruction"] for e in all_episodes]),
                    episode_lengths=np.array([e["num_frames"] for e in all_episodes], dtype=np.int32),
                )
                print(f"  💾 Checkpoint saved -> {npz_save_path} ({collected} episodes)")

    finally:
        for r in renderers.values():
            r.close()

    elapsed = time.time() - t0
    # Final NPZ packaging
    npz_save_path = out_path if out_path.suffix == ".npz" else (out_dir / "diffusion_300_dataset.npz")
    if all_episodes:
        np.savez_compressed(
            str(npz_save_path),
            obs_states=np.concatenate([e["obs_state"] for e in all_episodes], axis=0),
            actions=np.concatenate([e["action"] for e in all_episodes], axis=0),
            instructions=np.array([e["instruction"] for e in all_episodes]),
            episode_lengths=np.array([e["num_frames"] for e in all_episodes], dtype=np.int32),
        )

    meta = {
        "total_episodes": collected,
        "requested": args.count,
        "seed_start": args.seed_start,
        "record_hz": 20,
        "elapsed_s": round(elapsed, 2),
        "avg_sec_per_ep": round(elapsed / max(1, collected), 2),
        "cameras": ["overhead", "left_wrist", "right_wrist"],
        "npz_dataset": str(npz_save_path),
    }
    meta_path = out_dir / "metadata_300.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n{'═'*60}")
    print(f"  Dataset Generation Complete: {collected}/{args.count} episodes in {elapsed/60:.2f} minutes")
    print(f"  NPZ Dataset: {npz_save_path}")
    print(f"  Metadata:    {meta_path}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
