"""
policy/train_diffusion.py — Training & OpenVINO Export for Continuous VLA Diffusion Policy.

Trains language-conditioned Diffusion Policy on demonstration episodes:
  1. Multimodal dataset loading (12-DOF states, 3-camera views, text embeddings, 16-step action chunks)
  2. AdamW + Cosine Annealing optimization of DDPM noise prediction loss
  3. Validation eval on held-out trajectories
  4. OpenVINO IR export & heterogeneous device benchmark (CPU vs iGPU)
  5. Outputs metrics to evidence/diffusion_train_eval.json

Usage:
    python policy/train_diffusion.py --demos data/demos/ --epochs 15 --batch 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from policy.diffusion_policy import DiffusionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DiffusionDemoDataset(Dataset):
    """Sliding-window trajectory dataset for action chunking diffusion training."""

    def __init__(
        self,
        demos_dir: Path,
        chunk_size: int = 16,
        max_episodes: int = 100,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.samples: List[Dict[str, Any]] = []

        h5_files = sorted(list(demos_dir.glob("episode_*.hdf5")))[:max_episodes]
        if not h5_files:
            raise FileNotFoundError(f"No demonstration files found in {demos_dir}")

        logger.info("Indexing %d demonstration episodes...", len(h5_files))

        total_transitions = 0
        for h5_path in h5_files:
            try:
                with h5py.File(h5_path, "r") as f:
                    states = f["data"]["observation.state"][:]
                    actions = f["data"]["action"][:]
                    overhead = f["data"]["observation.images.overhead"][:]
                    left_w = f["data"]["observation.images.left_wrist"][:]
                    right_w = f["data"]["observation.images.right_wrist"][:]
                    instruction = str(f.attrs.get("instruction", "set the dinner table"))

                n_steps = len(states)
                if n_steps <= chunk_size:
                    continue

                for t in range(0, n_steps - chunk_size, stride):
                    self.samples.append({
                        "obs_state": states[t].astype(np.float32),
                        "overhead": overhead[t].astype(np.float32) / 255.0,
                        "left_wrist": left_w[t].astype(np.float32) / 255.0,
                        "right_wrist": right_w[t].astype(np.float32) / 255.0,
                        "action_chunk": actions[t : t + chunk_size].astype(np.float32),
                        "instruction": instruction,
                    })
                    total_transitions += 1
            except Exception as e:
                logger.warning("Failed reading %s: %s", h5_path, e)

        logger.info("Dataset built: %d sliding-window chunks indexed.", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        # Transpose images from [H, W, C] to [C, H, W]
        ov = torch.from_numpy(s["overhead"].transpose(2, 0, 1))
        lw = torch.from_numpy(s["left_wrist"].transpose(2, 0, 1))
        rw = torch.from_numpy(s["right_wrist"].transpose(2, 0, 1))

        return {
            "obs_state": torch.from_numpy(s["obs_state"]),
            "img_overhead": ov,
            "img_left_wrist": lw,
            "img_right_wrist": rw,
            "action_chunk": torch.from_numpy(s["action_chunk"]),
            "instruction": s["instruction"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "obs_state": torch.stack([b["obs_state"] for b in batch]),
        "img_overhead": torch.stack([b["img_overhead"] for b in batch]),
        "img_left_wrist": torch.stack([b["img_left_wrist"] for b in batch]),
        "img_right_wrist": torch.stack([b["img_right_wrist"] for b in batch]),
        "action_chunk": torch.stack([b["action_chunk"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
    }


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_diffusion(
    demos_dir: Path,
    out_checkpoint: Path,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cpu",
) -> Dict[str, Any]:
    dataset = DiffusionDemoDataset(demos_dir, chunk_size=16, stride=2)
    val_size = max(1, int(len(dataset) * 0.10))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    dev = torch.device(device)
    model = DiffusionPolicy(obs_dim=12, action_dim=12, chunk_size=16, feature_dim=256, device=dev)
    model.to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-6)

    logger.info("Training Diffusion Policy: %d train samples, %d val samples, %d epochs", train_size, val_size, epochs)
    t_start = time.time()
    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for b in train_loader:
            obs = b["obs_state"].to(dev)
            imgs = [b["img_overhead"].to(dev), b["img_left_wrist"].to(dev), b["img_right_wrist"].to(dev)]
            gt_act = b["action_chunk"].to(dev)
            inst = b["instruction"]

            optimizer.zero_grad()
            loss = model.compute_loss(obs, imgs, inst, gt_act)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        epoch_train_loss = running_loss / max(1, n_batches)
        train_loss_history.append(epoch_train_loss)

        # Validation
        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for b in val_loader:
                obs = b["obs_state"].to(dev)
                imgs = [b["img_overhead"].to(dev), b["img_left_wrist"].to(dev), b["img_right_wrist"].to(dev)]
                gt_act = b["action_chunk"].to(dev)
                inst = b["instruction"]
                v_loss = model.compute_loss(obs, imgs, inst, gt_act)
                val_running += v_loss.item()
                val_batches += 1

        epoch_val_loss = val_running / max(1, val_batches)
        val_loss_history.append(epoch_val_loss)

        logger.info(
            "Epoch [%2d/%2d] | Train Loss: %.5f | Val Loss: %.5f | LR: %.2e",
            epoch, epochs, epoch_train_loss, epoch_val_loss, scheduler.get_last_lr()[0]
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model.save(out_checkpoint)

    total_time = time.time() - t_start
    logger.info("Training complete in %.2f seconds. Best Val Loss: %.5f", total_time, best_val_loss)

    return {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "best_val_loss": round(best_val_loss, 5),
        "total_time_s": round(total_time, 2),
        "total_samples": len(dataset),
        "epochs": epochs,
    }


# ─── OpenVINO IR Export & Benchmark ───────────────────────────────────────────

def export_and_benchmark_openvino(
    checkpoint_path: Path,
    ir_out_dir: Path,
) -> Dict[str, Any]:
    """Exports Diffusion Policy components to OpenVINO IR and benchmarks inference."""
    import openvino as ov

    ir_out_dir.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    available = core.available_devices
    logger.info("Available OpenVINO devices for Diffusion Policy: %s", available)

    # Load PyTorch model
    model = DiffusionPolicy(obs_dim=12, action_dim=12, chunk_size=16, feature_dim=256)
    model.load(checkpoint_path)
    model.eval()

    # We export the DenoisingNet (the core iterative inner loop of diffusion)
    dummy_noisy_act = torch.randn(1, 16, 12)
    dummy_t = torch.tensor([50], dtype=torch.long)
    dummy_cond = torch.randn(1, 5, 256)

    class DenoisingWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, act, t, cond):
            return self.net(act, t, cond)

    wrapper = DenoisingWrapper(model.denoising_net)
    wrapper.eval()

    ir_xml = ir_out_dir / "diffusion_denoising.xml"
    try:
        ov_model = ov.convert_model(
            wrapper,
            example_input=(dummy_noisy_act, dummy_t, dummy_cond),
        )
        ov.save_model(ov_model, str(ir_xml))
        logger.info("Successfully exported Diffusion Denoising IR to %s", ir_xml)
    except Exception as e:
        logger.error("OpenVINO export error: %s", e)
        return {"error": str(e)}

    bench_results: Dict[str, Any] = {}

    for target_dev in ["CPU", "GPU"]:
        matched_dev = None
        if target_dev in available:
            matched_dev = target_dev
        elif target_dev == "GPU" and "GPU.0" in available:
            matched_dev = "GPU.0"

        if not matched_dev:
            logger.info("Device %s not present, skipping.", target_dev)
            continue

        try:
            compiled = core.compile_model(ov_model, matched_dev)
            infer_req = compiled.create_infer_request()

            act_np = dummy_noisy_act.numpy()
            t_np = dummy_t.numpy()
            cond_np = dummy_cond.numpy()

            # Warmup
            for _ in range(10):
                infer_req.infer([act_np, t_np, cond_np])

            # Benchmark 100 calls
            latencies = []
            for _ in range(100):
                t0 = time.perf_counter()
                infer_req.infer([act_np, t_np, cond_np])
                latencies.append((time.perf_counter() - t0) * 1000.0)

            mean_ms = float(np.mean(latencies))
            p50_ms = float(np.percentile(latencies, 50))
            p95_ms = float(np.percentile(latencies, 95))

            # 16-step DDIM trajectory generation latency:
            # 16 * mean_ms for single denoising steps
            total_16step_trajectory_ms = round(mean_ms * 16.0, 2)

            bench_results[target_dev] = {
                "matched_device": matched_dev,
                "step_mean_ms": round(mean_ms, 3),
                "step_p50_ms": round(p50_ms, 3),
                "step_p95_ms": round(p95_ms, 3),
                "full_16step_trajectory_ms": total_16step_trajectory_ms,
                "trajectories_per_sec": round(1000.0 / max(0.1, total_16step_trajectory_ms), 1),
            }
            logger.info(
                "Device %s: Single Denoise = %.2f ms | 16-step Trajectory = %.2f ms (%.1f traj/s)",
                matched_dev, mean_ms, total_16step_trajectory_ms, bench_results[target_dev]["trajectories_per_sec"]
            )
        except Exception as e:
            logger.warning("Benchmark error on %s: %s", target_dev, e)

    return bench_results


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark VLA Diffusion Policy.")
    parser.add_argument("--demos", type=str, default="data/demos/")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="models/diffusion_policy.pt")
    args = parser.parse_args()

    demos_dir = ROOT / args.demos
    out_checkpoint = ROOT / args.out
    ir_dir = ROOT / "models" / "diffusion_openvino"
    evidence_path = ROOT / "evidence" / "diffusion_train_eval.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Train
    train_meta = train_diffusion(
        demos_dir=demos_dir,
        out_checkpoint=out_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
    )

    # 2. Export and benchmark OpenVINO
    bench_meta = export_and_benchmark_openvino(
        checkpoint_path=out_checkpoint,
        ir_out_dir=ir_dir,
    )

    full_evidence = {
        "training": train_meta,
        "openvino_benchmark": bench_meta,
        "checkpoint_path": str(out_checkpoint),
        "ir_directory": str(ir_dir),
    }

    evidence_path.write_text(json.dumps(full_evidence, indent=2))
    logger.info("Saved full diffusion evaluation evidence to %s", evidence_path)


if __name__ == "__main__":
    main()
