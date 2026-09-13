"""
scripts/eval_seeds.py — Real multi-seed evaluation of robotic manipulation policies.

Evaluates policies across N deterministic domain-randomized seeds and logs
verified physical task metrics, sub-goal sequencing, and bimanual coordination.

Usage:
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

EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_policy(seeds: int, policy_name: str) -> Dict[str, Any]:
    xml_path = ROOT / "scene" / "dinner_table.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    results_per_seed: List[Dict[str, Any]] = []
    total_subgoals = 0
    total_tasks_success = 0
    bimanual_count = 0
    handoff_count = 0

    print(f"\n{'═'*60}")
    print(f"  Evaluating '{policy_name}' policy across {seeds} seeds")
    print(f"{'═'*60}")

    for seed in range(seeds):
        randomizer = DomainRandomizer(seed=seed)
        rand_log = randomizer.randomize(model, data)
        ctrl = ScriptedController(model, data, rand_log) if policy_name == "scripted" else None
        monitor = TaskMonitor(model, data)

        step = 0
        max_steps = 2200 if policy_name == "scripted" else 500

        while step < max_steps:
            if ctrl is not None:
                done = ctrl.step(model, data)
            else:
                # Negative baseline: hold home position
                done = False
                mujoco.mj_step(model, data)

            monitor.step(model, data)
            step += 1
            if done:
                break

        summary = monitor.get_summary()
        sg_count = sum(1 for v in summary.get("sub_goals", {}).values() if v)
        success = summary.get("task_success", False)

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
            "sub_goals": summary.get("sub_goals", {}),
        })

        status_str = "SUCCESS" if success else f"PARTIAL ({sg_count}/5)"
        print(f"  Seed {seed:2d}: {status_str} in {step:4d} steps | Sequencing: {summary.get('sequencing', 0)}/5")

    agg = {
        "policy": policy_name,
        "seeds_evaluated": seeds,
        "task_success_count": total_tasks_success,
        "task_success_rate": round(total_tasks_success / max(1, seeds), 4),
        "subgoals_achieved": total_subgoals,
        "total_possible_subgoals": seeds * 5,
        "subgoal_success_rate": round(total_subgoals / max(1, seeds * 5), 4),
        "bimanual_coordination_rate": round(bimanual_count / max(1, seeds), 4),
        "handoff_rate": round(handoff_count / max(1, seeds), 4),
        "seeds_detail": results_per_seed,
    }

    out_file = EVIDENCE_DIR / f"eval_seeds_{policy_name}.json"
    out_file.write_text(json.dumps(agg, indent=2))
    print(f"\n  Saved empirical evaluation -> {out_file}")
    print(f"  Summary: {total_tasks_success}/{seeds} Tasks | {total_subgoals}/{seeds * 5} Sub-Goals (Rate: {agg['subgoal_success_rate']*100:.1f}%)\n")
    return agg


def main():
    parser = argparse.ArgumentParser(description="Multi-seed physical policy evaluation.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--policy", type=str, default="scripted", choices=["scripted", "none"])
    args = parser.parse_args()

    evaluate_policy(args.seeds, args.policy)


if __name__ == "__main__":
    main()
