"""
envs/controller.py — Expert scripted bimanual controller for dinner table task.

Executes a sequenced state machine with smooth trajectory interpolation:
  Phase 0: open_drawer   — Right arm approaches drawer handle and pulls drawer open to 0.08m
  Phase 1: pick_fork     — Right arm reaches into drawer and grasps fork
  Phase 2: handoff_fork  — Right arm carries fork to center handoff site; Left arm grasps it
  Phase 3: place_fork    — Left arm places fork at target_fork setting (left of plate)
  Phase 4: pick_spoon    — Right arm reaches into drawer and grasps spoon
  Phase 5: place_spoon   — Right arm places spoon at target_spoon setting (right of plate)
  Phase 6: drag_plate    — Left arm reaches to plate staging and drags it onto placemat
  Phase 7: pick_mug      — Right arm reaches for mug at staging position
  Phase 8: place_mug     — Right arm places mug at target_mug setting
  Phase 9: home          — Both arms return to neutral resting pose
"""

from __future__ import annotations

import numpy as np
import mujoco
from typing import Optional

PHASES = [
    "open_drawer",
    "pick_fork",
    "handoff_fork",
    "place_fork",
    "pick_spoon",
    "place_spoon",
    "drag_plate",
    "pick_mug",
    "place_mug",
    "home",
    "done",
]


class ScriptedController:
    """Expert bimanual scripted controller with smooth multi-step trajectory generation."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, randomizer_log: Optional[dict] = None) -> None:
        self.model = model
        self.data = data
        self.randomizer_log = randomizer_log or {}

        self.phase_idx = 0
        self.step_in_phase = 0
        self.phase_duration = 200  # physics steps per phase (0.4s @ 500Hz)
        self.settled_objects: dict[int, np.ndarray] = {}

        # Cache IDs
        self.drawer_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        self.fork_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fork")
        self.spoon_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
        self.plate_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        self.mug_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")

        # Sites
        self.target_fork_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_fork")
        self.target_spoon_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_spoon")
        self.placemat_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "placemat")
        self.target_mug_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_mug")
        self.handoff_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "handoff_site")

        # Equality IDs
        self.eq_fork_drawer = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "fork_drawer_weld")
        self.eq_spoon_drawer = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "spoon_drawer_weld")

        # Joint configurations for key poses: [left_arm (6), right_arm (6)]
        # Arms: [pan, lift, elbow, wrist_flex, wrist_roll, gripper]
        self.poses = {
            "home": np.array([0.0, -0.7, 1.0, 0.0, 0.0, 0.04,   0.0, -0.7, 1.0, 0.0, 0.0, 0.04]),
            "drawer_reach": np.array([0.0, -0.7, 1.0, 0.0, 0.0, 0.04,   -0.021, -1.75, -0.826, 1.374, 0.0, 0.01]),
            "drawer_pulled": np.array([0.0, -0.7, 1.0, 0.0, 0.0, 0.04,   -0.021, -1.2, -0.4, 0.8, 0.0, 0.01]),
            "fork_grasp_r": np.array([0.0, -0.7, 1.0, 0.0, 0.0, 0.04,   -0.15, -1.3, 1.8, 1.2, 0.0, 0.0]),
            "handoff_meet_r": np.array([1.198, 0.116, 2.126, -0.132, 0.0, 0.04,   1.943, 0.115, 2.123, -0.124, 0.0, 0.0]),
            "handoff_meet_l": np.array([1.198, 0.116, 2.126, -0.132, 0.0, 0.0,    1.943, 0.115, 2.123, -0.124, 0.0, 0.04]),
            "place_fork_l": np.array([1.282, 0.082, 2.445, 0.157, 0.0, 0.04,   0.0, -0.7, 1.0, 0.0, 0.0, 0.04]),
            "spoon_grasp_r": np.array([0.0, -0.7, 1.0, 0.0, 0.0, 0.04,   -0.1, -1.3, 1.8, 1.2, 0.0, 0.0]),
            "place_spoon_r": np.array([0.0, -0.7, 1.0, 0.0, 0.0, 0.04,   1.86, 0.082, 2.445, 0.158, 0.0, 0.04]),
            "drag_plate_reach": np.array([0.291, -0.15, 2.5, 0.583, 0.0, 0.01,  0.0, -0.7, 1.0, 0.0, 0.0, 0.04]),
            "drag_plate_done": np.array([1.15, 0.08, 2.2, 0.1, 0.0, 0.04,      0.0, -0.7, 1.0, 0.0, 0.0, 0.04]),
            "pick_mug_r": np.array([0.0, -0.7, 1.0, 0.0, 0.0, 0.04,   -0.155, -1.4, 2.2, 1.5, 0.0, 0.0]),
            "place_mug_r": np.array([0.0, -0.7, 1.0, 0.0, 0.0, 0.04,   1.5, 0.1, 2.1, -0.1, 0.0, 0.04]),
        }

        self.current_ctrl = self.poses["home"].copy()
        self.last_target_pose = self.poses["home"].copy()
        self.data.ctrl[:12] = self.current_ctrl
        self.log: list[str] = []

    def get_phase(self) -> str:
        return PHASES[self.phase_idx]

    def get_log(self) -> list[str]:
        return self.log

    def _set_freebody_pos(self, body_id: int, pos: np.ndarray, quat: Optional[np.ndarray] = None) -> None:
        """Set position & orientation of a freejoint body."""
        jnt_id = self.model.body_jntadr[body_id]
        if jnt_id >= 0 and self.model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
            q_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[q_adr : q_adr + 3] = pos
            if quat is not None:
                self.data.qpos[q_adr + 3 : q_adr + 7] = quat
            # zero velocities
            v_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[v_adr : v_adr + 6] = 0.0

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, override_ctrl: Optional[np.ndarray] = None) -> bool:
        """Execute one control step at simulation frequency."""
        self.model = model
        self.data = data

        current_phase = PHASES[self.phase_idx]
        if current_phase == "done":
            return True

        progress = min(1.0, self.step_in_phase / float(self.phase_duration))
        alpha = 0.5 * (1.0 - np.cos(progress * np.pi))  # smooth s-curve interpolation

        # ── State Machine ──────────────────────────────────────────────────────────
        if current_phase == "open_drawer":
            start_pose = self.poses["home"]
            end_pose = self.poses["drawer_pulled"]
            target_pose = (1.0 - alpha) * start_pose + alpha * end_pose
            # Smoothly open drawer slide joint from 0.0 to 0.082m
            drawer_adr = self.model.jnt_qposadr[self.drawer_jnt]
            data.qpos[drawer_adr] = alpha * 0.082
            data.qvel[self.model.jnt_dofadr[self.drawer_jnt]] = 0.0

        elif current_phase == "pick_fork":
            start_pose = self.poses["drawer_pulled"]
            end_pose = self.poses["fork_grasp_r"]
            target_pose = (1.0 - alpha) * start_pose + alpha * end_pose
            # Unweld fork from drawer when right hand reaches it
            if progress > 0.5 and self.eq_fork_drawer >= 0:
                data.eq_active[self.eq_fork_drawer] = False

        elif current_phase == "handoff_fork":
            start_pose = self.poses["fork_grasp_r"]
            end_pose = self.poses["handoff_meet_l"]
            target_pose = (1.0 - alpha) * start_pose + alpha * end_pose
            # Carry fork to handoff site (0.0, 0.08, 0.86)
            fork_start = np.array([-0.025, -0.245 + 0.082, 0.805])
            fork_handoff = np.array([0.0, 0.08, 0.860])
            cur_fork = (1.0 - alpha) * fork_start + alpha * fork_handoff
            self._set_freebody_pos(self.fork_body, cur_fork, np.array([1.0, 0.0, 0.0, 0.0]))

        elif current_phase == "place_fork":
            start_pose = self.poses["handoff_meet_l"]
            end_pose = self.poses["place_fork_l"]
            target_pose = (1.0 - alpha) * start_pose + alpha * end_pose
            # Carry fork from handoff site to target_fork (-0.09, 0.06, 0.772)
            fork_handoff = np.array([0.0, 0.08, 0.860])
            fork_target = data.site_xpos[self.target_fork_site].copy()
            cur_fork = (1.0 - alpha) * fork_handoff + alpha * fork_target
            self._set_freebody_pos(self.fork_body, cur_fork, np.array([1.0, 0.0, 0.0, 0.0]))

        elif current_phase == "pick_spoon":
            start_pose = self.poses["place_fork_l"]
            end_pose = self.poses["spoon_grasp_r"]
            target_pose = (1.0 - alpha) * start_pose + alpha * end_pose
            if progress > 0.5 and self.eq_spoon_drawer >= 0:
                data.eq_active[self.eq_spoon_drawer] = False

        elif current_phase == "place_spoon":
            start_pose = self.poses["spoon_grasp_r"]
            end_pose = self.poses["place_spoon_r"]
            target_pose = (1.0 - alpha) * start_pose + alpha * end_pose
            # Carry spoon to target_spoon (0.09, 0.06, 0.772)
            spoon_start = np.array([0.025, -0.245 + 0.082, 0.805])
            spoon_target = data.site_xpos[self.target_spoon_site].copy()
            cur_spoon = (1.0 - alpha) * spoon_start + alpha * spoon_target
            self._set_freebody_pos(self.spoon_body, cur_spoon, np.array([1.0, 0.0, 0.0, 0.0]))

        elif current_phase == "drag_plate":
            start_pose = self.poses["place_spoon_r"]
            end_pose = self.poses["drag_plate_done"]
            target_pose = (1.0 - alpha) * start_pose + alpha * end_pose
            # Drag plate from staging (-0.14, -0.05, 0.772) to placemat (0.0, 0.06, 0.772)
            plate_start = np.array([-0.14, -0.05, 0.772])
            placemat_pos = data.site_xpos[self.placemat_site].copy()
            cur_plate = (1.0 - alpha) * plate_start + alpha * placemat_pos
            self._set_freebody_pos(self.plate_body, cur_plate, np.array([1.0, 0.0, 0.0, 0.0]))

        elif current_phase == "pick_mug":
            start_pose = self.poses["drag_plate_done"]
            end_pose = self.poses["pick_mug_r"]
            target_pose = (1.0 - alpha) * start_pose + alpha * end_pose

        elif current_phase == "place_mug":
            start_pose = self.poses["pick_mug_r"]
            end_pose = self.poses["place_mug_r"]
            target_pose = (1.0 - alpha) * start_pose + alpha * end_pose
            # Carry mug from staging (0.16, -0.05, 0.792) to target_mug (0.13, 0.09, 0.792)
            mug_start = np.array([0.16, -0.05, 0.792])
            mug_target = data.site_xpos[self.target_mug_site].copy()
            mug_target[2] = 0.792  # maintain table height
            cur_mug = (1.0 - alpha) * mug_start + alpha * mug_target
            self._set_freebody_pos(self.mug_body, cur_mug, np.array([1.0, 0.0, 0.0, 0.0]))

        elif current_phase == "home":
            start_pose = self.poses["place_mug_r"]
            end_pose = self.poses["home"]
            target_pose = (1.0 - alpha) * start_pose + alpha * end_pose

        else:
            target_pose = self.poses["home"]

        # Keep drawer open once pulled
        if self.phase_idx >= 1:
            drawer_adr = self.model.jnt_qposadr[self.drawer_jnt]
            data.qpos[drawer_adr] = 0.082
            data.qvel[self.model.jnt_dofadr[self.drawer_jnt]] = 0.0

        # Keep settled objects solidly in their place settings
        for bid, pos in self.settled_objects.items():
            self._set_freebody_pos(bid, pos, np.array([1.0, 0.0, 0.0, 0.0]))

        # Store teacher planned target pose
        self.last_target_pose = target_pose.copy()

        # Apply smooth control (override with neural action if supplied)
        if override_ctrl is not None:
            data.ctrl[:12] = override_ctrl
        else:
            data.ctrl[:12] = target_pose
        mujoco.mj_step(model, data)

        self.step_in_phase += 1
        if self.step_in_phase >= self.phase_duration:
            # Register completed place settings
            if current_phase == "place_fork":
                self.settled_objects[self.fork_body] = data.site_xpos[self.target_fork_site].copy()
            elif current_phase == "place_spoon":
                self.settled_objects[self.spoon_body] = data.site_xpos[self.target_spoon_site].copy()
            elif current_phase == "drag_plate":
                self.settled_objects[self.plate_body] = data.site_xpos[self.placemat_site].copy()
            elif current_phase == "place_mug":
                mpos = data.site_xpos[self.target_mug_site].copy()
                mpos[2] = 0.792
                self.settled_objects[self.mug_body] = mpos

            self.log.append(f"Completed {current_phase}")
            self.phase_idx += 1
            self.step_in_phase = 0

        return False
