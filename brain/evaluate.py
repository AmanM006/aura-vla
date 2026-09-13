"""
brain/evaluate.py — End-to-end evaluation over N random seeds.

Usage:
    python brain/evaluate.py --seeds 10                          # scripted controller
    python brain/evaluate.py --seeds 10 --hard                  # harder randomization
    python brain/evaluate.py --seeds 10 --policy openvino       # learned policy via OV
    python brain/evaluate.py --seeds 10 --precision int8        # specific OV precision
    python brain/evaluate.py --seeds 5 --bump plate --video     # recovery demo
"""

from __future__ import annotations

import argparse
import json
import time
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# Defensive guard: osmesa is Linux/WSL/Docker only; Windows uses native OpenGL (or egl)
if sys.platform == "win32" and os.environ.get("MUJOCO_GL") in ("osmesa", "OSMESA"):
    del os.environ["MUJOCO_GL"]

EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)


def run_episode(
    seed: int,
    policy: str = "scripted",
    hard: bool = False,
    precision: str = "fp32",
    bump_object: Optional[str] = None,
    record_video: bool = False,
) -> dict:
    """Run a single evaluation episode and return a results dict."""
    import mujoco

    xml_path = ROOT / "scene" / "dinner_table.xml"
    if not xml_path.exists():
        raise FileNotFoundError(
            f"Scene XML not found at {xml_path}. "
            "Run `python scene/build_scene.py --headless-build` first."
        )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Reset to home
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Randomize scene
    try:
        from envs.randomize import DomainRandomizer
        rng_seed = seed + (1000 if hard else 0)
        randomizer = DomainRandomizer(seed=rng_seed)
        rand_log = randomizer.randomize(model, data)
    except Exception as e:
        print(f"    [WARN] Randomizer failed: {e} — using nominal scene")
        rand_log = {}

    # Task monitor
    try:
        from envs.task import TaskMonitor
        monitor = TaskMonitor(model, data)
    except Exception as e:
        raise RuntimeError(f"TaskMonitor init failed: {e}") from e

    # Optionally bump an object mid-task (recovery demo)
    bump_step = None
    if bump_object:
        bump_step = 500  # bump at step 500

    # Load policy
    if policy == "scripted":
        try:
            from envs.controller import ScriptedController
            ctrl = ScriptedController(model, data, rand_log)
        except Exception as e:
            raise RuntimeError(f"ScriptedController init failed: {e}") from e
        policy_fn = lambda m, d, step: ctrl.step(m, d)
        get_phase = lambda: ctrl.get_phase()
    elif policy == "openvino":
        try:
            from policy.openvino_runtime import ACTOpenVINORuntime
            from brain.planner import VLMPlanner
            ov_rt = ACTOpenVINORuntime(
                model_dir=ROOT / "models" / "act_openvino",
                device="CPU",
                precision=precision,
            )
            planner = VLMPlanner(device="GPU")
            plan = planner.plan("set the table", {})
            plan_idx = [0]

            def policy_fn(m, d, step):
                if plan_idx[0] >= len(plan):
                    return True  # done
                obs_state = np.array(d.qpos[:12], dtype=np.float32)
                # Collect camera images
                renderer = mujoco.Renderer(m, height=128, width=128)
                images = []
                for cam_name in ["top_cam", "left_wrist_cam", "right_wrist_cam"]:
                    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                    renderer.update_scene(d, camera=cam_id)
                    images.append(renderer.render().copy())
                renderer.close()
                task_token = plan_idx[0]
                actions = ov_rt.predict(obs_state, images, task_token)
                # Apply first action
                data.ctrl[:] = actions[:model.nu]
                mujoco.mj_step(m, d)
                return False

            get_phase = lambda: f"plan[{plan_idx[0]}]/{len(plan)}"
        except Exception as e:
            print(f"    [WARN] OpenVINO policy unavailable: {e}. Falling back to scripted.")
            from envs.controller import ScriptedController
            ctrl = ScriptedController(model, data, rand_log)
            policy_fn = lambda m, d, step: ctrl.step(m, d)
            get_phase = lambda: ctrl.get_phase()
    elif policy == "none":
        # Arms hold home pose — negative control
        policy_fn = lambda m, d, step: (mujoco.mj_step(m, d), False)[1]
        get_phase = lambda: "none"
    else:
        raise ValueError(f"Unknown policy: {policy}")

    # Video recorder
    frames = []

    # ── Run episode ──
    t0 = time.time()
    max_steps = 15000  # 30 seconds at 500 Hz
    done = False

    for step in range(max_steps):
        # Optional mid-task bump (recovery demo)
        if bump_step and step == bump_step and bump_object:
            try:
                obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bump_object)
                if obj_id >= 0:
                    data.xfrc_applied[obj_id, :3] = [0.5, 0.3, 0.0]
            except Exception:
                pass
        if bump_step and step == bump_step + 50 and bump_object:
            # Remove the force
            try:
                obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bump_object)
                if obj_id >= 0:
                    data.xfrc_applied[obj_id, :3] = [0.0, 0.0, 0.0]
            except Exception:
                pass

        # Step physics
        try:
            done = policy_fn(model, data, step)
        except Exception as e:
            print(f"    [WARN] Policy step failed at step {step}: {e}")
            break

        # Update task monitor
        monitor.step(model, data)

        # Record frame (every 5 steps = ~10 fps for video)
        if record_video and step % 5 == 0:
            try:
                renderer = mujoco.Renderer(model, height=240, width=320)
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam")
                renderer.update_scene(data, camera=cam_id)
                frames.append(renderer.render().copy())
                renderer.close()
            except Exception:
                pass

        if done:
            break

    elapsed = time.time() - t0
    summary = monitor.get_summary()

    result = {
        "seed": seed,
        "policy": policy,
        "precision": precision,
        "hard": hard,
        "steps_run": step + 1,
        "elapsed_s": round(elapsed, 2),
        "bump_object": bump_object,
        **summary,
        "randomizer_log": rand_log,
    }

    # Save video if requested
    if record_video and frames:
        try:
            import imageio
            video_path = EVIDENCE_DIR / f"demo_seed{seed}_{policy}.mp4"
            imageio.mimwrite(str(video_path), frames, fps=10, quality=8)
            result["video_path"] = str(video_path)
            print(f"    Video saved: {video_path}")
        except Exception as e:
            print(f"    [WARN] Video save failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate the hackathon pipeline over N seeds.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of evaluation seeds (0..N-1)")
    parser.add_argument("--policy", choices=["scripted", "openvino", "none"], default="scripted")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("--hard", action="store_true", help="Harder randomization (±4cm starts)")
    parser.add_argument("--bump", type=str, default=None, metavar="OBJECT",
                        help="Bump this object mid-task to test recovery (e.g. 'plate')")
    parser.add_argument("--video", action="store_true", help="Record episode video")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  AI Infra Summit Hackathon — Evaluation")
    print(f"  policy={args.policy}  seeds={args.seeds}  hard={args.hard}")
    print(f"{'═'*60}\n")

    results = []
    sub_goal_totals = {
        "drawer_open": 0, "fork_placed": 0, "spoon_placed": 0,
        "plate_placed": 0, "mug_placed": 0,
    }
    task_success_count = 0

    for seed in range(args.seeds):
        print(f"  Seed {seed:2d} / {args.seeds - 1}  ", end="", flush=True)
        try:
            r = run_episode(
                seed=seed,
                policy=args.policy,
                hard=args.hard,
                precision=args.precision,
                bump_object=args.bump,
                record_video=args.video,
            )
            results.append(r)

            sub_goals = r.get("sub_goals", {})
            achieved = sum(1 for v in sub_goals.values() if v)
            task_ok = r.get("task_success", False)
            if task_ok:
                task_success_count += 1

            for k in sub_goal_totals:
                if sub_goals.get(k, False):
                    sub_goal_totals[k] += 1

            status_str = "  ".join(
                f"{'✅' if sub_goals.get(k, False) else '❌'} {k.split('_')[0]}"
                for k in ["drawer_open", "fork_placed", "spoon_placed", "plate_placed", "mug_placed"]
            )
            result_icon = "🏆" if task_ok else "🔄"
            print(f"{result_icon}  {achieved}/5  |  {status_str}  |  {r['elapsed_s']:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"seed": seed, "error": str(e)})

    # ── Summary ──
    total_sub_goals = sum(sub_goal_totals.values())
    max_sub_goals = args.seeds * 5

    print(f"\n{'═'*60}")
    print(f"  RESULTS — policy={args.policy}  seeds={args.seeds}")
    print(f"{'═'*60}")
    print(f"  Sub-goals:     {total_sub_goals} / {max_sub_goals}  ({100*total_sub_goals/max_sub_goals:.1f}%)")
    print(f"  Task success:  {task_success_count} / {args.seeds}")
    print()
    for k, v in sub_goal_totals.items():
        bar = "█" * v + "░" * (args.seeds - v)
        print(f"  {k:15s}: {v:2d}/{args.seeds}  [{bar}]")
    print()

    # Save evidence
    evidence = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": args.policy,
        "precision": args.precision,
        "seeds": args.seeds,
        "hard": args.hard,
        "total_sub_goals": total_sub_goals,
        "max_sub_goals": max_sub_goals,
        "task_success_count": task_success_count,
        "sub_goal_totals": sub_goal_totals,
        "episodes": results,
    }
    label = args.policy
    if args.hard:
        label += "_hard"
    out = EVIDENCE_DIR / f"eval_seeds_{label}.json"
    out.write_text(json.dumps(evidence, indent=2))
    print(f"  Evidence saved to {out}\n")


if __name__ == "__main__":
    main()
