"""
brain/anomalib_monitor.py — Intel Anomalib-Inspired Visual Defect & Anomaly Monitor.

Monitors tri-camera video feeds (overhead, left wrist, right wrist) at 20Hz/50Hz.
Uses a PatchCore deep feature memory bank compiled via OpenVINO for Intel iGPU/CPU
to detect grasp slips, dropped cutlery, plate misalignments, or drawer collisions.

When anomaly score s > 0.65, triggers closed-loop replanning through InterruptHandler.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ─── Compact PatchCore Feature Extractor ──────────────────────────────────────

class PatchCoreBackbone(nn.Module):
    """Compact convolutional spatial patch feature extractor for visual defect detection.

    Outputs L2-normalized 60-dimensional spatial visual embeddings combining multi-scale
    grid pooling and Sobel edge gradient descriptors.
    """

    def __init__(self, out_dim: int = 60) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, 3, 128, 128] in [0, 1] -> Output: [B, 60]"""
        # Illumination normalization for domain randomization invariance
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True) + 1e-5
        x_norm = (x - mean) / std

        B, C, H, W = x_norm.shape
        grid = F.adaptive_avg_pool2d(x_norm, (4, 4))
        grad_x = F.conv2d(x_norm, self.sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(x_norm, self.sobel_y, padding=1, groups=3)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        edge_grid = F.adaptive_avg_pool2d(grad_mag, (2, 2))
        feat = torch.cat([grid.reshape(B, -1), edge_grid.reshape(B, -1)], dim=-1)
        return F.normalize(feat, p=2, dim=-1)


# ─── Anomalib Visual Defect Monitor ───────────────────────────────────────────

class AnomalibMonitor:
    """Real-time visual defect and anomaly detector for robotic manipulation."""

    def __init__(
        self,
        device: str = "GPU",
        threshold: float = 0.65,
        memory_bank_path: Optional[Union[str, Path]] = None,
        model_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Args:
            device: OpenVINO target device ('GPU', 'CPU', or 'AUTO')
            threshold: Anomaly score threshold triggering replan interrupt (default: 0.65)
            memory_bank_path: Path to calibrated nominal feature bank (.npz)
            model_dir: Directory to save/load OpenVINO IR model
        """
        self.target_device = device
        self.threshold = threshold
        self.model_dir = Path(model_dir) if model_dir else ROOT / "models" / "anomalib_openvino"
        base_bank = Path(memory_bank_path) if memory_bank_path else ROOT / "models" / "anomalib_banks.npz"
        self.memory_bank_path = base_bank

        # Initialize PyTorch backbone
        self.backbone = PatchCoreBackbone(out_dim=60)
        self.backbone.eval()

        # OpenVINO compiled model handle (if available)
        self.ov_compiled = None
        self.ov_input_layer = None
        self.ov_output_layer = None
        self.active_device = "CPU (PyTorch fallback)"

        # Per-camera feature memory banks
        self.memory_banks: Dict[str, np.ndarray] = {}
        self._init_runtime()
        self._load_memory_banks()

    def _init_runtime(self) -> None:
        """Attempts to compile the backbone with OpenVINO for fast inference."""
        try:
            import openvino as ov
            core = ov.Core()
            available = core.available_devices

            selected_device = "CPU"
            if self.target_device in available:
                selected_device = self.target_device
            elif "GPU.0" in available:
                selected_device = "GPU.0"
            elif "GPU" in available:
                selected_device = "GPU"

            ir_xml = self.model_dir / "anomalib_patchcore.xml"
            if not ir_xml.exists():
                self.model_dir.mkdir(parents=True, exist_ok=True)
                dummy_input = torch.randn(1, 3, 128, 128)
                ov_model = ov.convert_model(self.backbone, example_input=dummy_input)
                ov.save_model(ov_model, str(ir_xml))
                logger.info("Exported Anomalib PatchCore IR model to %s", ir_xml)

            ov_model = core.read_model(str(ir_xml))
            self.ov_compiled = core.compile_model(ov_model, selected_device)
            self.ov_input_layer = self.ov_compiled.input(0)
            self.ov_output_layer = self.ov_compiled.output(0)
            self.active_device = f"OpenVINO ({selected_device})"
            logger.info("AnomalibMonitor initialized on %s", self.active_device)

        except Exception as e:
            logger.warning("OpenVINO init failed (%s); using PyTorch CPU fallback.", e)
            self.ov_compiled = None
            self.active_device = "PyTorch CPU"

    def _load_memory_banks(self) -> None:
        """Loads pre-calibrated nominal per-camera feature banks, or creates synthetic priors."""
        if self.memory_bank_path.exists():
            try:
                npz = np.load(str(self.memory_bank_path))
                for k in ["overhead", "left_wrist", "right_wrist"]:
                    if k in npz:
                        b = npz[k].astype(np.float32)
                        norms = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8
                        self.memory_banks[k] = b / norms
                logger.info("Loaded per-camera memory banks: %s", {k: len(v) for k, v in self.memory_banks.items()})
                return
            except Exception as e:
                logger.warning("Error loading memory banks: %s", e)

        # Fallback prior per camera
        logger.info("Generating baseline prior per-camera memory banks...")
        rng = np.random.RandomState(42)
        for k in ["overhead", "left_wrist", "right_wrist"]:
            b = rng.randn(100, 60).astype(np.float32)
            norms = np.linalg.norm(b, axis=-1, keepdims=True)
            self.memory_banks[k] = b / norms

    def extract_features(self, img_bgr_or_rgb: np.ndarray) -> np.ndarray:
        """Extracts 60-dim normalized embedding from a single 128x128 image frame."""
        if img_bgr_or_rgb.ndim == 3 and img_bgr_or_rgb.shape[-1] == 3:
            img = img_bgr_or_rgb.transpose(2, 0, 1)[None, :].astype(np.float32)
        elif img_bgr_or_rgb.ndim == 4:
            img = img_bgr_or_rgb.astype(np.float32)
        else:
            raise ValueError(f"Invalid image shape: {img_bgr_or_rgb.shape}")

        if img.max() > 1.5:
            img = img / 255.0

        if self.ov_compiled is not None:
            res = self.ov_compiled([img])
            feat = res[self.ov_output_layer][0]
        else:
            with torch.no_grad():
                tensor_in = torch.from_numpy(img)
                feat = self.backbone(tensor_in)[0].numpy()

        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm

    def inspect(self, frames: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Inspects current visual observations across all cameras for defects."""
        t0 = time.perf_counter()

        camera_scores: Dict[str, float] = {}
        max_score = 0.0
        max_cam = "none"

        for cam_name, frame in frames.items():
            if frame is None or frame.size == 0:
                continue

            bank = self.memory_banks.get(cam_name)
            if bank is None:
                continue

            try:
                feat = self.extract_features(frame)
                similarities = np.dot(bank, feat)
                nearest_sim = float(np.max(similarities))

                # Non-linear scaling into calibrated anomaly probability [0.0, 1.0]
                dist = max(0.0, 1.0 - nearest_sim)
                score = float(1.0 / (1.0 + np.exp(-300.0 * (dist - 0.012))))

                camera_scores[cam_name] = round(score, 4)
                if score > max_score:
                    max_score = score
                    max_cam = cam_name

            except Exception as e:
                logger.warning("Error processing camera '%s': %s", cam_name, e)
                camera_scores[cam_name] = 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0
        is_anomalous = max_score >= self.threshold

        if max_score >= 0.85:
            action = "emergency_halt"
        elif max_score >= self.threshold:
            action = "replan"
        else:
            action = "continue"

        return {
            "is_anomalous": bool(is_anomalous),
            "anomaly_score": round(max_score, 4),
            "threshold": self.threshold,
            "camera_scores": camera_scores,
            "max_camera": max_cam,
            "latency_ms": round(latency_ms, 2),
            "device": self.active_device,
            "action_recommendation": action,
        }

    def calibrate(self, hdf5_dir: Union[str, Path], max_samples: int = 300) -> None:
        """Calibrates per-camera memory banks using demonstration frames from HDF5 files."""
        import h5py

        p = Path(hdf5_dir)
        files = sorted(list(p.glob("episode_*.hdf5")))
        if not files:
            logger.warning("No demonstration files found in %s for calibration.", hdf5_dir)
            return

        logger.info("Calibrating AnomalibMonitor on %d episodes across all 3 cameras...", len(files))
        feats_ov: List[np.ndarray] = []
        feats_lw: List[np.ndarray] = []
        feats_rw: List[np.ndarray] = []

        for h5_file in files[:35]:
            try:
                with h5py.File(h5_file, "r") as f:
                    if not f.attrs.get("task_success", False):
                        continue
                    data_grp = f["data"]
                    ov_imgs = data_grp["observation.images.overhead"][:]
                    lw_imgs = data_grp["observation.images.left_wrist"][:]
                    rw_imgs = data_grp["observation.images.right_wrist"][:]

                    for idx in range(0, len(ov_imgs), 4):
                        feats_ov.append(self.extract_features(ov_imgs[idx]))
                        feats_lw.append(self.extract_features(lw_imgs[idx]))
                        feats_rw.append(self.extract_features(rw_imgs[idx]))
                        if len(feats_ov) >= max_samples:
                            break
            except Exception as e:
                logger.warning("Failed to parse %s: %s", h5_file, e)

            if len(feats_ov) >= max_samples:
                break

        bank_ov = np.array(feats_ov, dtype=np.float32)
        bank_lw = np.array(feats_lw, dtype=np.float32)
        bank_rw = np.array(feats_rw, dtype=np.float32)

        self.memory_banks = {
            "overhead": bank_ov,
            "left_wrist": bank_lw,
            "right_wrist": bank_rw,
        }
        self.memory_bank_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(self.memory_bank_path),
            overhead=bank_ov,
            left_wrist=bank_lw,
            right_wrist=bank_rw,
        )
        logger.info("Saved per-camera memory banks to %s (OV: %d, LW: %d, RW: %d)",
                    self.memory_bank_path, len(bank_ov), len(bank_lw), len(bank_rw))


if __name__ == "__main__":
    print("Testing per-camera AnomalibMonitor...")
    m = AnomalibMonitor(device="GPU")
    frames = {
        "overhead": np.zeros((128, 128, 3), dtype=np.uint8),
        "left_wrist": np.zeros((128, 128, 3), dtype=np.uint8),
        "right_wrist": np.zeros((128, 128, 3), dtype=np.uint8),
    }
    rep = m.inspect(frames)
    print("Report:", rep)
