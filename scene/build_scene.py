"""
scene/build_scene.py -- Builds the dual SO-101 dinner table MJCF scene.

Usage:
    python scene/build_scene.py                   # generates XML + opens viewer
    python scene/build_scene.py --headless-build  # generates XML only, no viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCENE_DIR = Path(__file__).parent
SCENE_DIR.mkdir(exist_ok=True)


def make_arm_xml(side: str, x: float, yaw_deg: float) -> str:
    """Generate MJCF XML fragment for one SO-101 arm. No f-string expressions
    in attribute values (MuJoCo doesn't evaluate Python) -- all values are
    pre-computed strings."""
    s = side
    cam_euler = "90 0 0" if s == "left" else "-90 0 0"
    jaw_rgba  = "1 0 0 1" if s == "left" else "0 0 1 1"

    return f"""
        <!-- {s.upper()} ARM -->
        <body name="{s}_arm_base" pos="{x} 0 0.76" euler="0 0 {yaw_deg}">
            <inertial pos="0 0 0" mass="0.5" diaginertia="0.01 0.01 0.01"/>
            <geom name="{s}_base_geom" type="cylinder" size="0.025 0.025" rgba="0.3 0.3 0.3 1"/>

            <body name="{s}_link1" pos="0 0 0.045">
                <joint name="{s}_shoulder_pan" type="hinge" axis="0 0 1"
                       range="-2.8 2.8" damping="0.5" armature="0.05"/>
                <inertial pos="0 0 0.03" mass="0.3" diaginertia="0.005 0.005 0.005"/>
                <geom type="capsule" size="0.020 0.040" rgba="0.4 0.4 0.8 1"/>

                <body name="{s}_link2" pos="0 0 0.08">
                    <joint name="{s}_shoulder_lift" type="hinge" axis="0 1 0"
                           range="-1.8 1.8" damping="0.5" armature="0.05"/>
                    <inertial pos="0 0 0.05" mass="0.3" diaginertia="0.005 0.005 0.005"/>
                    <geom type="capsule" size="0.018 0.060" rgba="0.4 0.4 0.8 1"/>

                    <body name="{s}_link3" pos="0 0 0.125">
                        <joint name="{s}_elbow_flex" type="hinge" axis="0 1 0"
                               range="-2.5 2.5" damping="0.3" armature="0.03"/>
                        <inertial pos="0 0 0.05" mass="0.2" diaginertia="0.003 0.003 0.003"/>
                        <geom type="capsule" size="0.016 0.055" rgba="0.5 0.5 0.9 1"/>

                        <body name="{s}_link4" pos="0 0 0.115">
                            <joint name="{s}_wrist_flex" type="hinge" axis="0 1 0"
                                   range="-2.5 2.5" damping="0.1" armature="0.01"/>
                            <inertial pos="0 0 0.03" mass="0.1" diaginertia="0.001 0.001 0.001"/>
                            <geom type="capsule" size="0.014 0.030" rgba="0.6 0.6 0.9 1"/>

                            <body name="{s}_link5" pos="0 0 0.065">
                                <joint name="{s}_wrist_roll" type="hinge" axis="0 0 1"
                                       range="-3.14 3.14" damping="0.05" armature="0.005"/>
                                <inertial pos="0 0 0.02" mass="0.08" diaginertia="0.0005 0.0005 0.0005"/>
                                <geom type="capsule" size="0.013 0.020" rgba="0.7 0.7 0.9 1"/>

                                <body name="{s}_gripper_base" pos="0 0 0.045">
                                    <inertial pos="0 0 0.01" mass="0.05" diaginertia="0.0002 0.0002 0.0002"/>
                                    <geom type="box" size="0.025 0.015 0.012" rgba="0.2 0.2 0.2 1"/>
                                    <!-- Wrist camera: pos/euler only, no site attribute -->
                                    <camera name="{s}_wrist_cam" pos="0 0.02 0.015" euler="{cam_euler}" fovy="60"/>
                                    <!-- Fixed jaw -->
                                    <body name="{s}_fixed_jaw" pos="0 -0.015 0.020">
                                        <inertial pos="0 0 0.01" mass="0.02" diaginertia="0.0001 0.0001 0.0001"/>
                                        <geom name="{s}_fixed_jaw_geom" type="box" size="0.008 0.004 0.022"
                                              rgba="0.8 0.2 0.2 1" contype="1" conaffinity="1" friction="1.5 0.1 0.1"/>
                                    </body>
                                    <!-- Moving jaw -->
                                    <body name="{s}_moving_jaw" pos="0 0.015 0.020">
                                        <joint name="{s}_gripper_joint" type="slide" axis="0 1 0"
                                               range="0 0.04" damping="0.5" armature="0.001"/>
                                        <inertial pos="0 0 0.01" mass="0.02" diaginertia="0.0001 0.0001 0.0001"/>
                                        <geom name="{s}_moving_jaw_geom" type="box" size="0.008 0.004 0.022"
                                              rgba="0.2 0.8 0.2 1" contype="1" conaffinity="1" friction="1.5 0.1 0.1"/>
                                    </body>
                                    <!-- Jaw meeting-point site (41 mm forward of wrist) -->
                                    <site name="{s}_jaw_site"   pos="0 0 0.041" size="0.005" rgba="{jaw_rgba}"/>
                                    <site name="{s}_grasp_site" pos="0 0 0.050" size="0.003" rgba="1 1 0 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>"""


def build_scene_xml() -> str:
    """Return the full dinner table MJCF as a string."""
    left_arm  = make_arm_xml("left",  -0.22,  18.0)
    right_arm = make_arm_xml("right",  0.22, -18.0)

    return f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="dinner_table">

    <compiler angle="radian" autolimits="true"/>

    <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast">
        <flag contact="enable" frictionloss="enable"/>
    </option>

    <default>
        <joint limited="true"/>
        <geom contype="1" conaffinity="1" friction="1.0 0.05 0.01"
              solimp="0.9 0.95 0.001" solref="0.01 1"/>
        <motor ctrllimited="true"/>
    </default>

    <asset>
        <material name="table_mat"   rgba="0.72 0.55 0.35 1"/>
        <material name="plate_mat"   rgba="0.92 0.92 0.88 1"/>
        <material name="mug_mat"     rgba="0.85 0.4  0.3  1"/>
        <material name="bottle_mat"  rgba="0.3  0.6  0.3  1"/>
        <material name="cutlery_mat" rgba="0.75 0.75 0.75 1"/>
        <material name="cabinet_mat" rgba="0.6  0.45 0.25 1"/>
        <material name="placemat_mat" rgba="0.3 0.5  0.8  1"/>
        <material name="floor_mat"   rgba="0.55 0.55 0.55 1"/>
    </asset>

    <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" material="floor_mat"/>

        <!-- Lighting -->
        <light name="key_light" pos="0.3 -0.5 2.0" dir="-0.2 0.4 -1"
               directional="true" diffuse="0.85 0.85 0.85" specular="0.3 0.3 0.3"/>
        <light name="fill_light" pos="-0.5 0.5 1.5" dir="0.3 -0.3 -1"
               directional="true" diffuse="0.35 0.35 0.35"/>

        <!-- Cameras (world-attached) -->
        <camera name="top_cam"  pos="0 0 1.45"    euler="0 0 0"   fovy="55"/>
        <camera name="front_cam" pos="0 -0.7 1.05" euler="25 0 0"  fovy="50"/>
        <camera name="op_cam"   pos="0.5 -0.6 1.1" euler="20 0 -30" fovy="50"/>

        <!-- TABLE -->
        <body name="table" pos="0 0 0">
            <geom name="table_top" type="box" pos="0 0 0.755" size="0.44 0.28 0.015"
                  material="table_mat" mass="20"/>
            <geom type="box" pos="-0.38 -0.22 0.38" size="0.02 0.02 0.38" rgba="0.4 0.3 0.2 1"/>
            <geom type="box" pos=" 0.38 -0.22 0.38" size="0.02 0.02 0.38" rgba="0.4 0.3 0.2 1"/>
            <geom type="box" pos="-0.38  0.22 0.38" size="0.02 0.02 0.38" rgba="0.4 0.3 0.2 1"/>
            <geom type="box" pos=" 0.38  0.22 0.38" size="0.02 0.02 0.38" rgba="0.4 0.3 0.2 1"/>
        </body>

        <!-- CABINET WITH DRAWER -->
        <body name="cabinet" pos="0 -0.245 0.77">
            <geom name="cab_back"  type="box" pos="0 0.005 0.06" size="0.085 0.003 0.060" material="cabinet_mat" mass="2"/>
            <geom name="cab_left"  type="box" pos="-0.085 0.03 0.06" size="0.003 0.030 0.060" material="cabinet_mat"/>
            <geom name="cab_right" type="box" pos=" 0.085 0.03 0.06" size="0.003 0.030 0.060" material="cabinet_mat"/>
            <geom name="cab_top"   type="box" pos="0 0.03 0.120" size="0.085 0.030 0.003" material="cabinet_mat"/>
            <geom name="cab_base"  type="box" pos="0 0.03 0.000" size="0.085 0.030 0.003" material="cabinet_mat"/>

            <body name="drawer" pos="0 0 0.030">
                <joint name="drawer_slide" type="slide" axis="0 1 0"
                       range="0 0.09" damping="5.0" armature="0.001"/>
                <geom name="drawer_front"     type="box" pos="0 -0.025 0"  size="0.080 0.003 0.025" material="cabinet_mat" mass="0.2"/>
                <geom name="drawer_back_wall" type="box" pos="0  0.025 0"  size="0.080 0.003 0.025" material="cabinet_mat"/>
                <geom name="drawer_lwall"     type="box" pos="-0.080 0 0"  size="0.003 0.028 0.025" material="cabinet_mat"/>
                <geom name="drawer_rwall"     type="box" pos=" 0.080 0 0"  size="0.003 0.028 0.025" material="cabinet_mat"/>
                <geom name="drawer_floor"     type="box" pos="0 0 -0.025"  size="0.080 0.028 0.003" material="cabinet_mat"/>
                <geom name="drawer_handle"    type="capsule" pos="0 -0.030 0.015" size="0.006 0.030"
                      euler="90 0 0" rgba="0.6 0.6 0.6 1" mass="0.05"/>
                <site name="drawer_handle_site" pos="0 -0.030 0.015" size="0.010"/>
                <!-- Fork and spoon sites (reference positions) -->
                <site name="fork_in_drawer"  pos="-0.025 0 0.005" size="0.005" rgba="0.8 0.8 0 0.5"/>
                <site name="spoon_in_drawer" pos=" 0.025 0 0.005" size="0.005" rgba="0.8 0.8 0 0.5"/>
            </body>
        </body>

        <!-- TABLE OBJECTS -->
        <!-- Fork & Spoon: world-level free bodies, welded to drawer initially -->
        <!-- Cabinet at y=-0.245, drawer body at z=0.77+0.03=0.80, drawer starts closed (y_offset=0) -->
        <body name="fork" pos="-0.025 -0.245 0.805">
            <freejoint name="fork_joint"/>
            <inertial pos="0 0 0" mass="0.025" diaginertia="0.00002 0.00002 0.000002"/>
            <geom name="fork_geom" type="capsule" size="0.004 0.060" euler="0 90 0"
                  rgba="0.75 0.75 0.75 1" friction="1.0 0.1 0.01" contype="3" conaffinity="3"/>
            <site name="fork_site" pos="0 0 0" size="0.006"/>
        </body>

        <body name="spoon" pos="0.025 -0.245 0.805">
            <freejoint name="spoon_joint"/>
            <inertial pos="0 0 0" mass="0.020" diaginertia="0.00002 0.00002 0.000002"/>
            <geom name="spoon_geom" type="capsule" size="0.004 0.055" euler="0 90 0"
                  rgba="0.75 0.75 0.75 1" friction="1.0 0.1 0.01" contype="3" conaffinity="3"/>
            <site name="spoon_site" pos="0 0 0" size="0.006"/>
        </body>
        <!-- Plate (staging on table left, to be placed on placemat) -->
        <body name="plate" pos="-0.14 -0.05 0.772">
            <freejoint name="plate_joint"/>
            <inertial pos="0 0 0" mass="0.3" diaginertia="0.0004 0.0004 0.0008"/>
            <geom name="plate_base" type="cylinder" size="0.055 0.003"
                  material="plate_mat" friction="0.8 0.05 0.01" contype="3" conaffinity="3"/>
            <geom name="plate_rim"  type="cylinder" size="0.055 0.005" pos="0 0 0.006"
                  material="plate_mat" friction="0.8 0.05 0.01" contype="3" conaffinity="3"/>
            <site name="plate_site" pos="0 0 0" size="0.008"/>
        </body>

        <!-- Mug (staging on table right, to be placed at target_mug) -->
        <body name="mug" pos="0.16 -0.05 0.792">
            <freejoint name="mug_joint"/>
            <inertial pos="0 0 0.04" mass="0.25" diaginertia="0.0002 0.0002 0.0002"/>
            <geom name="mug_wall" type="cylinder" size="0.035 0.040"
                  material="mug_mat" friction="0.8 0.05 0.01" contype="3" conaffinity="3"/>
            <site name="mug_site" pos="0 0 0.04" size="0.008"/>
        </body>

        <body name="bottle" pos="-0.18 0.12 0.862">
            <freejoint name="bottle_joint"/>
            <inertial pos="0 0 0.1" mass="0.5" diaginertia="0.0003 0.0003 0.0001"/>
            <geom name="bottle_geom" type="cylinder" size="0.025 0.100"
                  material="bottle_mat" friction="0.8 0.05 0.01" contype="3" conaffinity="3"/>
            <site name="bottle_site" pos="0 0 0.1" size="0.006"/>
        </body>

        <!-- Placemat indicator (no collision) -->
        <body name="placemat_body" pos="0 0.06 0.771">
            <geom name="placemat_geom" type="box" size="0.075 0.050 0.001"
                  material="placemat_mat" contype="0" conaffinity="0" mass="0"/>
        </body>

        <!-- TARGET SITES for scoring predicates -->
        <site name="target_fork"  pos="-0.09 0.06 0.772" size="0.008" rgba="0 1 0 0.5"/>
        <site name="target_spoon" pos=" 0.09 0.06 0.772" size="0.008" rgba="0 0 1 0.5"/>
        <site name="placemat"     pos=" 0.00 0.06 0.772" size="0.010" rgba="0.3 0.5 0.8 0.5"/>
        <site name="target_mug"   pos=" 0.13 0.09 0.772" size="0.008" rgba="1 0 0 0.5"/>
        <site name="handoff_site" pos=" 0.00 0.08 0.860" size="0.010" rgba="1 1 0 0.5"/>

        <!-- ROBOT ARMS -->
{left_arm}
{right_arm}

    </worldbody>

    <!-- ACTUATORS: 12 position-controlled joints (6 per arm) -->
    <actuator>
        <position name="left_shoulder_pan"  joint="left_shoulder_pan"  kp="120" ctrlrange="-2.8 2.8"/>
        <position name="left_shoulder_lift" joint="left_shoulder_lift" kp="120" ctrlrange="-1.8 1.8"/>
        <position name="left_elbow_flex"    joint="left_elbow_flex"    kp="80"  ctrlrange="-2.5 2.5"/>
        <position name="left_wrist_flex"    joint="left_wrist_flex"    kp="40"  ctrlrange="-2.5 2.5"/>
        <position name="left_wrist_roll"    joint="left_wrist_roll"    kp="20"  ctrlrange="-3.14 3.14"/>
        <position name="left_gripper"       joint="left_gripper_joint" kp="80"  ctrlrange="0 0.04"/>
        <position name="right_shoulder_pan"  joint="right_shoulder_pan"  kp="120" ctrlrange="-2.8 2.8"/>
        <position name="right_shoulder_lift" joint="right_shoulder_lift" kp="120" ctrlrange="-1.8 1.8"/>
        <position name="right_elbow_flex"    joint="right_elbow_flex"    kp="80"  ctrlrange="-2.5 2.5"/>
        <position name="right_wrist_flex"    joint="right_wrist_flex"    kp="40"  ctrlrange="-2.5 2.5"/>
        <position name="right_wrist_roll"    joint="right_wrist_roll"    kp="20"  ctrlrange="-3.14 3.14"/>
        <position name="right_gripper"       joint="right_gripper_joint" kp="80"  ctrlrange="0 0.04"/>
    </actuator>

    <!-- EQUALITY CONSTRAINTS -->
    <equality>
        <!-- Fork and spoon start welded to drawer (controller releases these when it opens drawer) -->
        <weld name="fork_drawer_weld"  active="true"  body1="drawer" body2="fork"
              relpose="-0.025 0 0.005  1 0 0 0" solref="0.005 1" solimp="0.99 0.999 0.0001"/>
        <weld name="spoon_drawer_weld" active="true"  body1="drawer" body2="spoon"
              relpose=" 0.025 0 0.005  1 0 0 0" solref="0.005 1" solimp="0.99 0.999 0.0001"/>
        <!-- Grasp welds (inactive initially, controller activates these to grasp objects) -->
        <weld name="left_grasp_weld"  active="false" body1="left_gripper_base"  body2="fork"
              relpose="0 0 0.05  1 0 0 0" solref="0.005 1" solimp="0.99 0.999 0.0001"/>
        <weld name="right_grasp_weld" active="false" body1="right_gripper_base" body2="spoon"
              relpose="0 0 0.05  1 0 0 0" solref="0.005 1" solimp="0.99 0.999 0.0001"/>
        <weld name="left_grasp_weld2"  active="false" body1="left_gripper_base"  body2="mug"
              relpose="0 0 0.05  1 0 0 0" solref="0.005 1" solimp="0.99 0.999 0.0001"/>
        <weld name="right_grasp_weld2" active="false" body1="right_gripper_base" body2="plate"
              relpose="0 0 0.05  1 0 0 0" solref="0.005 1" solimp="0.99 0.999 0.0001"/>
    </equality>

    <!-- KEYFRAME: arm joints only (freejoint bodies use their body pos declaration) -->
    <keyframe>
        <key name="home" ctrl="0 0 0 0 0 0.04  0 0 0 0 0 0.04"/>
    </keyframe>

</mujoco>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dinner table MuJoCo scene.")
    parser.add_argument("--headless-build", action="store_true",
                        help="Skip interactive viewer (Docker/CI mode)")
    args = parser.parse_args()

    print("Building dinner table scene XML...")
    xml = build_scene_xml()

    out_path = SCENE_DIR / "dinner_table.xml"
    out_path.write_text(xml, encoding="utf-8")
    print(f"  Saved -> {out_path}")

    # Verify it loads
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_string(xml)
        data  = mujoco.MjData(model)
        print(f"  Loaded OK: nq={model.nq}  nbody={model.nbody}  nu={model.nu}  ncam={model.ncam}")

        # 1-second stability check
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        for _ in range(500):
            mujoco.mj_step(model, data)
        max_qvel = float(abs(data.qvel).max())
        print(f"  Stability (1s): max|qvel| = {max_qvel:.6f}")

    except Exception as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    if not args.headless_build:
        try:
            import mujoco.viewer
            print("  Opening interactive viewer (close window to exit)...")
            model2 = mujoco.MjModel.from_xml_path(str(out_path))
            data2  = mujoco.MjData(model2)
            mujoco.viewer.launch(model2, data2)
        except Exception as exc:
            print(f"  Viewer unavailable ({exc}) -- headless assumed.")

    print("Done.")


if __name__ == "__main__":
    main()
