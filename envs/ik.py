"""
envs/ik.py — Damped-least-squares IK solver specifically designed for dual SO-101 arms.

Targets the jaw meeting-point site (41mm forward of wrist), isolating each arm's
5 revolute joints while respecting joint range limits.
"""

from __future__ import annotations

import numpy as np
import mujoco
from typing import Tuple, Optional


def solve_arm_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: str,  # 'left' or 'right'
    target_pos: np.ndarray,
    damping: float = 0.02,
    max_iter: int = 150,
    tol: float = 1e-4,
    standoff_m: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Solves inverse kinematics for a single SO-101 arm (6 values: 5 joints + gripper).

    Returns:
        arm_ctrl: np.ndarray of shape (6,)
        residual_mm: float (position error in millimeters)
    """
    site_name = f"{arm}_jaw_site"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site {site_name} not found in model")

    # Arm joint DOFs in model
    # SO-101 joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
    jnt_names = [
        f"{arm}_shoulder_pan",
        f"{arm}_shoulder_lift",
        f"{arm}_elbow_flex",
        f"{arm}_wrist_flex",
        f"{arm}_wrist_roll",
    ]
    dof_indices = []
    qpos_indices = []
    joint_ranges = []

    for name in jnt_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_indices.append(model.jnt_dofadr[jid])
        qpos_indices.append(model.jnt_qposadr[jid])
        joint_ranges.append(model.jnt_range[jid])

    dof_indices = np.array(dof_indices, dtype=int)
    qpos_indices = np.array(qpos_indices, dtype=int)
    joint_ranges = np.array(joint_ranges)

    # Save original qpos state to restore or iterate
    qpos_copy = data.qpos.copy()
    current_arm_qpos = qpos_copy[qpos_indices].copy()

    # If standoff is requested, offset target along negative jaw axis
    actual_target = target_pos.copy()
    if standoff_m != 0.0:
        actual_target[2] += standoff_m

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    residual = 1.0
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[site_id]
        err_pos = actual_target - current_pos
        residual = float(np.linalg.norm(err_pos))
        if residual < tol:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        j_sub = jacp[:, dof_indices]

        # DLS update
        delta_q = j_sub.T @ np.linalg.solve(j_sub @ j_sub.T + (damping ** 2) * np.eye(3), err_pos)
        current_arm_qpos += delta_q * 0.5
        current_arm_qpos = np.clip(current_arm_qpos, joint_ranges[:, 0], joint_ranges[:, 1])
        data.qpos[qpos_indices] = current_arm_qpos

    # 6th value is gripper (default open = 0.04)
    arm_ctrl = np.zeros(6)
    arm_ctrl[:5] = current_arm_qpos
    arm_ctrl[5] = 0.04

    # Restore data qpos
    data.qpos[:] = qpos_copy
    mujoco.mj_kinematics(model, data)

    return arm_ctrl, residual * 1000.0


def plan_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: str,
    target_pos: np.ndarray,
    standoff_m: float = 0.04,
) -> Tuple[np.ndarray, float]:
    """Convenience wrapper for arm pose planning with standoff backoff."""
    return solve_arm_ik(model, data, arm, target_pos, standoff_m=standoff_m)
