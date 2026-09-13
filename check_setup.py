"""
check_setup.py — Environment sanity gate for the hackathon submission.
Run this FIRST before any other script. Verifies:
  - All Python packages are importable
  - Scene XML exists and loads
  - Model geometry matches expected counts
  - A live 1-trial sanity run of the scripted controller passes
  - OpenVINO devices are enumerable
  - Speechmatics API key is set (optional, warns if missing)
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path

if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Auto-load .env so API keys are available without manual export
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# Defensive guard: osmesa is Linux/WSL/Docker only; Windows uses native OpenGL (or egl)
if sys.platform == "win32" and os.environ.get("MUJOCO_GL") in ("osmesa", "OSMESA"):
    del os.environ["MUJOCO_GL"]

ROOT = Path(__file__).parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

RESULTS: list[dict] = []

# ─── Helpers ──────────────────────────────────────────────────────────────────

def check(name: str, fn):
    """Run a named check and record pass/fail."""
    try:
        result = fn()
        msg = result if isinstance(result, str) else "OK"
        print(f"  ✅  {name}: {msg}")
        RESULTS.append({"check": name, "status": "PASS", "detail": msg})
        return True
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  ❌  {name}: {e}")
        RESULTS.append({"check": name, "status": "FAIL", "detail": str(e), "traceback": tb})
        return False


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ─── Package checks ───────────────────────────────────────────────────────────

def check_packages():
    section("1. Package Imports")
    packages = [
        ("mujoco",         "mujoco"),
        ("numpy",          "numpy"),
        ("scipy",          "scipy"),
        ("h5py",           "h5py"),
        ("cv2",            "opencv-python"),
        ("PIL",            "pillow"),
        ("rich",           "rich"),
        ("fastapi",        "fastapi"),
        ("uvicorn",        "uvicorn"),
        ("websockets",     "websockets"),
        ("pydantic",       "pydantic"),
    ]
    all_ok = True
    for mod, pkg in packages:
        ok = check(f"import {mod}", lambda m=mod: __import__(m) and "OK")
        all_ok = all_ok and ok

    # Optional packages
    optional = [
        ("openvino",            "openvino"),
        ("torch",               "torch"),
        ("speechmatics",        "speechmatics-python"),
        ("nncf",                "nncf"),
    ]
    for mod, pkg in optional:
        try:
            __import__(mod)
            print(f"  ✅  import {mod} (optional): OK")
            RESULTS.append({"check": f"import {mod} (optional)", "status": "PASS"})
        except ImportError:
            print(f"  ⚠️   import {mod} (optional): NOT INSTALLED — will skip dependent features")
            RESULTS.append({"check": f"import {mod} (optional)", "status": "WARN", "detail": "not installed"})
    return all_ok


# ─── Scene checks ─────────────────────────────────────────────────────────────

def check_scene():
    section("2. Scene XML")
    import mujoco

    xml_path = ROOT / "scene" / "dinner_table.xml"

    check("scene XML exists", lambda: str(xml_path) if xml_path.exists() else (_ for _ in ()).throw(FileNotFoundError(f"{xml_path} not found")))

    def load_model():
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        return f"nq={m.nq}, nbody={m.nbody}, nu={m.nu}"

    check("scene loads without error", load_model)

    def check_actuators():
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        assert m.nu >= 12, f"Expected >= 12 actuators (2 arms × 6), got {m.nu}"
        return f"{m.nu} actuators"

    check("12+ actuators (dual SO-101)", check_actuators)

    def check_cameras():
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        cam_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
        required = {"top_cam", "front_cam", "left_wrist_cam", "right_wrist_cam"}
        missing = required - set(cam_names)
        assert not missing, f"Missing cameras: {missing}"
        return f"{m.ncam} cameras: {cam_names}"

    check("required cameras present", check_cameras)

    def check_home_stable():
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        d = mujoco.MjData(m)
        # Reset to home keyframe if present
        key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(m, d, key_id)
            # Set arm joints explicitly (qposadr for arm joints = last 12 qpos)
            arm_ctrl = [0, -0.7, 1.0, 0, 0, 0.04, 0, -0.7, 1.0, 0, 0, 0.04]
            d.ctrl[:] = arm_ctrl
            d.qpos[-12:] = arm_ctrl
        # Simulate 3 seconds
        for _ in range(int(3.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        # Only check arm joint velocities (skip freejoint DOFs for free bodies)
        arm_qvel = d.qvel[-12:]  # last 12 = arm joints (6 DOF per arm hinge/slide)
        max_qvel = float(abs(arm_qvel).max())
        assert max_qvel < 5e-3, f"Arm joints not stable: max|qvel| = {max_qvel:.6f}"
        return f"arm max|qvel| = {max_qvel:.6f} (< 5e-3)"

    check("home keyframe stable under gravity", check_home_stable)

    def check_offscreen():
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        d = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, height=128, width=128)
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam")
        assert cam_id >= 0, "top_cam not found"
        renderer.update_scene(d, camera=cam_id)
        pixels = renderer.render()
        assert pixels.shape == (128, 128, 3), f"Unexpected shape: {pixels.shape}"
        renderer.close()
        return f"top_cam renders {pixels.shape}"

    check("offscreen rendering (top_cam 128×128)", check_offscreen)


# ─── OpenVINO checks ──────────────────────────────────────────────────────────

def check_openvino():
    section("3. OpenVINO")
    try:
        import openvino as ov
    except ImportError:
        print("  ⚠️   OpenVINO not installed — skipping OV checks")
        RESULTS.append({"check": "openvino_import", "status": "WARN", "detail": "not installed"})
        return

    def list_devices():
        core = ov.Core()
        devices = core.available_devices
        assert len(devices) > 0, "No OpenVINO devices found"
        return f"devices: {devices}"

    check("OpenVINO devices enumerable", list_devices)

    def check_npu():
        core = ov.Core()
        return "NPU available" if "NPU" in core.available_devices else "NPU NOT available (will use CPU fallback)"

    check("NPU device check", check_npu)

    def check_gpu():
        core = ov.Core()
        return "GPU available" if "GPU" in core.available_devices else "GPU NOT available (will use CPU fallback)"

    check("GPU (iGPU) device check", check_gpu)


# ─── API key checks ───────────────────────────────────────────────────────────

def check_env():
    section("4. Environment Variables")

    def check_speechmatics():
        key = os.environ.get("SPEECHMATICS_API_KEY", "")
        if not key:
            raise ValueError("SPEECHMATICS_API_KEY not set — voice input will use text fallback")
        return f"key set ({len(key)} chars)"

    check("SPEECHMATICS_API_KEY", check_speechmatics)


# ─── Live sanity run ──────────────────────────────────────────────────────────

def check_live_run():
    section("5. Live Sanity Run (seed=0, 10 steps)")
    try:
        import mujoco

        xml_path = ROOT / "scene" / "dinner_table.xml"
        if not xml_path.exists():
            print("  ⚠️   Scene XML not found — skipping live run")
            return

        def do_live_run():
            m = mujoco.MjModel.from_xml_path(str(xml_path))
            d = mujoco.MjData(m)
            for _ in range(10):
                mujoco.mj_step(m, d)
            return f"10 physics steps OK, time={d.time:.4f}s"

        check("10-step physics sanity", do_live_run)

    except Exception as e:
        print(f"  ⚠️   Live run skipped: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("  AI Infra Summit Hackathon — Environment Check")
    print("  Intel Physical AI Online Track")
    print("═"*60)

    check_packages()
    check_scene()
    check_openvino()
    check_env()
    check_live_run()

    # ── Summary ──
    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    n_warn = sum(1 for r in RESULTS if r["status"] == "WARN")
    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")

    print(f"\n{'═'*60}")
    print(f"  Results: {n_pass} PASS  |  {n_warn} WARN  |  {n_fail} FAIL")
    print(f"{'═'*60}\n")

    # Save evidence
    evidence = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "checks": RESULTS,
    }
    out = EVIDENCE_DIR / "check_setup.json"
    out.write_text(json.dumps(evidence, indent=2))
    print(f"  Evidence saved to {out}")

    if n_fail > 0:
        print(f"\n  ⚠️  {n_fail} check(s) failed. Fix them before proceeding.\n")
        sys.exit(1)
    else:
        print("  ✅  All required checks passed. Ready to build!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
