import sys
from pathlib import Path
import cv2
import numpy as np
import mujoco
import subprocess

ROOT = Path(r"c:\Users\cheer\Documents\lab\hackathon")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.controller import ScriptedController
from envs.randomize import DomainRandomizer
from envs.task import TaskMonitor

def create_cam(lookat, dist, elev, azim):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = lookat
    cam.distance = dist
    cam.elevation = elev
    cam.azimuth = azim
    return cam

def record_all():
    xml_path = ROOT / "scene" / "dinner_table.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=480, width=640)

    # Seed 3
    randomizer = DomainRandomizer(seed=3)
    rand_log = randomizer.randomize(model, data)
    mujoco.mj_forward(model, data)

    ctrl = ScriptedController(model, data, rand_log)
    monitor = TaskMonitor(model, data)

    out_dir = ROOT / "dashboard-ui" / "public" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = {
        "front": create_cam([0.0, -0.05, 0.80], 1.35, -18.0, 90.0),
        "overhead": create_cam([0.0, 0.0, 0.78], 1.20, -65.0, 90.0),
        "left_wrist": create_cam([-0.10, 0.05, 0.80], 0.95, -25.0, 140.0),
        "right_wrist": create_cam([0.10, 0.05, 0.80], 0.95, -25.0, 40.0),
    }

    writers = {}
    temp_files = {}
    for name in cameras:
        temp_path = out_dir / f"temp_{name}.mp4"
        temp_files[name] = temp_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers[name] = cv2.VideoWriter(str(temp_path), fourcc, 25.0, (640, 480))

    step = 0
    max_steps = 2200
    render_interval = 8
    frames_recorded = 0

    print("Rendering 4 synchronized video angles...")

    while step < max_steps:
        done = ctrl.step(model, data)
        monitor.step(model, data)

        # Skip first 25 steps (~1.0s trim matching demo.mp4)
        if step >= 24 and step % render_interval == 0:
            for name, cam in cameras.items():
                renderer.update_scene(data, camera=cam)
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writers[name].write(bgr)
            frames_recorded += 1

        step += 1
        if done:
            print(f"Episode finished at step {step}!")
            for _ in range(20):
                for name, cam in cameras.items():
                    renderer.update_scene(data, camera=cam)
                    rgb = renderer.render()
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    writers[name].write(bgr)
                frames_recorded += 1
            break

    for w in writers.values():
        w.release()

    print(f"Recorded {frames_recorded} frames. Transcoding to H.264...")

    for name in cameras:
        src = temp_files[name]
        dest = out_dir / f"{name}.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level:v", "4.0",
            "-movflags", "+faststart",
            str(dest)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        src.unlink(missing_ok=True)
        print(f"[OK] {dest.name} ready ({dest.stat().st_size / 1024:.1f} KB)")

    print("All multi-angle videos generated!")

if __name__ == "__main__":
    record_all()
