"""
scripts/eval_seeds.py — Multi-seed physical policy evaluation.

Evaluates policies across N deterministic domain-randomized seeds:
  1. 'scripted'           — Heuristic inverse kinematics baseline (upper bound)
  2. 'diffusion'          — Continuous VLA Diffusion Policy (PyTorch) with Temporal Ensembling
  3. 'diffusion_openvino' — OpenVINO INT8/FP32 quantized Diffusion Policy
  4. 'none'               — Negative baseline (hold home pose)

Usage:
    python scripts/eval_seeds.py --seeds 10 --policy diffusion_openvino
    python scripts/eval_seeds.py --seeds 10 --policy scripted
    python scripts/eval_seeds.py --seeds 10 --policy none
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import mujoco
import numpy as np

if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32" and os.environ.get("MUJOCO_GL") in ("osmesa", "OSMESA"):
    del os.environ["MUJOCO_GL"]

from envs.controller import ScriptedController
from envs.randomize import DomainRandomizer
from envs.task import TaskMonitor
from policy.temporal_ensemble import TemporalEnsemble

EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_policy(seeds: int, policy_name: str, device: str = "CPU") -> Dict[str, Any]:
    xml_path = ROOT / "scene" / "dinner_table.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    results_per_seed: List[Dict[str, Any]] = []
    total_subgoals = 0
    total_tasks_success = 0
    bimanual_count = 0
    handoff_count = 0
    tracking_errors_mm = []

    print(f"\n{'═'*65}")
    print(f"  AURA-VLA Physical Evaluation: '{policy_name}' across {seeds} seeds")
    print(f"{'═'*65}")

    # Load neural policy if requested
    diffusion_policy = None
    ensemble = None
    renderers = None
    cam_ids = None

    if policy_name in ("diffusion", "diffusion_openvino"):
        ckpt_path = ROOT / "models" / "diffusion_policy.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}. Train the policy first.")

        import torch
        from policy.diffusion_policy import DiffusionPolicy

        dev = torch.device("cpu")
        diffusion_policy = DiffusionPolicy(obs_dim=12, action_dim=12, chunk_size=16, feature_dim=256, device=dev)
        diffusion_policy.load(ckpt_path)
        diffusion_policy.eval()

        if policy_name == "diffusion_openvino":
            ir_path = ROOT / "models" / "diffusion_openvino" / "diffusion_denoising.xml"
            if ir_path.exists():
                loaded = diffusion_policy.load_openvino(ir_path, device=device)
                if loaded:
                    print(f"  [OpenVINO Engine] Loaded compiled Heterogeneous IR on {device}")

        ensemble = TemporalEnsemble(action_dim=12, chunk_size=16, discount_m=0.05, query_stride=4)

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
        print(f"  [Neural Policy] Loaded Continuous Diffusion Policy & Temporal Ensemble (H=16, k=4)")

    shadow_data = mujoco.MjData(model)
    left_jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_jaw_site")
    right_jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_jaw_site")
    steps_per_policy = 25  # 500Hz sim / 20Hz policy = 25 physics steps per action

    try:
        for seed in range(seeds):
            randomizer = DomainRandomizer(seed=seed)
            rand_log = randomizer.randomize(model, data)
            ctrl_scripted = ScriptedController(model, data, rand_log)
            monitor = TaskMonitor(model, data)

            if ensemble:
                ensemble.reset()

            step = 0
            max_steps = 2200 if policy_name != "none" else 500
            seed_drift_mm = []
            current_action = ctrl_scripted.poses["home"].copy()

            while step < max_steps:
                if policy_name in ("diffusion", "diffusion_openvino"):
                    # 20Hz Policy Query & Trajectory Chunking
                    if step % steps_per_policy == 0:
                        if ensemble.needs_query():
                            obs_state = data.qpos[-12:].copy()
                            renderers["overhead"].update_scene(data, camera=cam_ids["overhead"])
                            renderers["left_wrist"].update_scene(data, camera=cam_ids["left_wrist"])
                            renderers["right_wrist"].update_scene(data, camera=cam_ids["right_wrist"])

                            imgs = [
                                renderers["overhead"].render().copy(),
                                renderers["left_wrist"].render().copy(),
                                renderers["right_wrist"].render().copy(),
                            ]
                            action_chunk = diffusion_policy.predict_action(
                                obs_state=obs_state,
                                images=imgs,
                                instruction="set the dinner table",
                                denoise_steps=16,
                            )
                        else:
                            action_chunk = None

                        current_action = ensemble.step(action_chunk)

                    # Step environment using neural policy action override
                    done = ctrl_scripted.step(model, data, override_ctrl=current_action)

                    # Forward Kinematics End-Effector Cartesian Error in mm
                    shadow_data.qpos[-12:] = ctrl_scripted.last_target_pose
                    mujoco.mj_kinematics(model, shadow_data)
                    err_l = float(np.linalg.norm(data.site_xpos[left_jaw_id] - shadow_data.site_xpos[left_jaw_id]) * 1000.0)
                    err_r = float(np.linalg.norm(data.site_xpos[right_jaw_id] - shadow_data.site_xpos[right_jaw_id]) * 1000.0)
                    seed_drift_mm.append((err_l + err_r) / 2.0)

                elif policy_name == "scripted":
                    done = ctrl_scripted.step(model, data)
                else:
                    # Negative baseline: hold home pose
                    done = False
                    mujoco.mj_step(model, data)
                    shadow_data.qpos[-12:] = ctrl_scripted.last_target_pose
                    mujoco.mj_kinematics(model, shadow_data)
                    err_l = float(np.linalg.norm(data.site_xpos[left_jaw_id] - shadow_data.site_xpos[left_jaw_id]) * 1000.0)
                    err_r = float(np.linalg.norm(data.site_xpos[right_jaw_id] - shadow_data.site_xpos[right_jaw_id]) * 1000.0)
                    seed_drift_mm.append((err_l + err_r) / 2.0)

                monitor.step(model, data)
                step += 1
                if done:
                    break

            summary = monitor.get_summary()
            sg_count = sum(1 for v in summary.get("sub_goals", {}).values() if v)
            success = summary.get("task_success", False)
            mean_drift = float(np.mean(seed_drift_mm)) if seed_drift_mm else 0.0
            tracking_errors_mm.append(mean_drift)

            total_subgoals += sg_count
            if success:
                total_tasks_success += 1
            if summary.get("bimanual", False):
                bimanual_count += 1
            if summary.get("handoff_occurred", False):
                handoff_count += 1

            results_per_seed.append({
                "seed": seed,
                "steps": step,
                "subgoals_passed": sg_count,
                "task_success": success,
                "sequencing": summary.get("sequencing", 0),
                "bimanual": summary.get("bimanual", False),
                "handoff": summary.get("handoff_occurred", False),
                "mean_drift_mm": round(mean_drift, 3),
                "sub_goals": summary.get("sub_goals", {}),
            })

            drift_tag = f" | Drift: {mean_drift:4.2f}mm" if seed_drift_mm else ""
            status_str = "SUCCESS" if success else f"PARTIAL ({sg_count}/5)"
            print(f"  Seed {seed:2d}: {status_str} in {step:4d} steps | Sequencing: {summary.get('sequencing', 0)}/5{drift_tag}")

    finally:
        if renderers:
            for r in renderers.values():
                r.close()

    agg = {
        "policy": policy_name,
        "device": device,
        "seeds_evaluated": seeds,
        "task_success_count": total_tasks_success,
        "task_success_rate": round(total_tasks_success / max(1, seeds), 4),
        "subgoals_achieved": total_subgoals,
        "total_possible_subgoals": seeds * 5,
        "subgoal_success_rate": round(total_subgoals / max(1, seeds * 5), 4),
        "bimanual_coordination_rate": round(bimanual_count / max(1, seeds), 4),
        "handoff_rate": round(handoff_count / max(1, seeds), 4),
        "mean_trajectory_tracking_error_mm": round(float(np.mean(tracking_errors_mm)), 3) if tracking_errors_mm else 0.0,
        "seeds_detail": results_per_seed,
    }

    out_file = EVIDENCE_DIR / f"eval_seeds_{policy_name}.json"
    out_file.write_text(json.dumps(agg, indent=2))
    print(f"\n{'═'*65}")
    print(f"  Saved Empirical Evaluation -> {out_file}")
    print(f"  Tasks: {total_tasks_success}/{seeds} ({agg['task_success_rate']*100:.1f}%) | Sub-Goals: {total_subgoals}/{seeds * 5} ({agg['subgoal_success_rate']*100:.1f}%)")
    if tracking_errors_mm and any(e > 0 for e in tracking_errors_mm):
        print(f"  Mean Neural Tracking Error: {agg['mean_trajectory_tracking_error_mm']:.2f} mm")
    print(f"{'═'*65}\n")
    return agg


def main():
    parser = argparse.ArgumentParser(description="Multi-seed physical policy evaluation.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--policy", type=str, default="scripted", choices=["scripted", "diffusion", "diffusion_openvino", "none"])
    parser.add_argument("--device", type=str, default="CPU")
    args = parser.parse_args()

    evaluate_policy(args.seeds, args.policy, args.device)


if __name__ == "__main__":
    main()