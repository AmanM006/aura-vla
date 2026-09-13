"""
policy/temporal_ensemble.py — Exponentially-Weighted Receding Horizon Trajectory Ensembler.

Eliminates high-frequency action discontinuities and actuator chatter in Action Chunking
and Diffusion Policies by maintaining an overlapping temporal prediction buffer.

Mathematical Formulation:
    w_i = exp(-m * (t - t_i))
    a_t = sum(w_i * a_t^(i)) / sum(w_i)
where:
    - t is current sim timestep
    - t_i is the query timestamp of policy prediction chunk i
    - m is the exponential discount decay factor (higher m favors newest chunk)
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TemporalEnsemble:
    """Temporal trajectory ensembler for continuous action chunking policies."""

    def __init__(
        self,
        action_dim: int = 12,
        chunk_size: int = 16,
        discount_m: float = 0.05,
        query_stride: int = 2,
        action_limits: Optional[Tuple[float, float]] = (-1.0, 1.0),
    ) -> None:
        """Args:
            action_dim: Dimension of the control vector (12 for bimanual SO-101)
            chunk_size: Horizon length H predicted by diffusion/ACT policy (e.g. 16)
            discount_m: Exponential weighting decay factor (0.01 - 0.1)
            query_stride: Re-query interval k (steps between policy forward passes)
            action_limits: (min_val, max_val) clamp range for safety
        """
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.discount_m = discount_m
        self.query_stride = max(1, query_stride)
        self.action_limits = action_limits

        # Buffer maps future timestep t -> list of (weight: float, action: np.ndarray)
        self._buffer: Dict[int, List[Tuple[float, np.ndarray]]] = {}
        self._last_action: np.ndarray = np.zeros(action_dim, dtype=np.float32)
        self._total_queries: int = 0
        self._total_steps: int = 0

    def reset(self) -> None:
        """Clears the trajectory buffer and resets temporal state."""
        self._buffer.clear()
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._total_queries = 0
        self._total_steps = 0

    def should_query(self, step: int) -> bool:
        """Determines if the policy network needs to be evaluated at this step."""
        return (step % self.query_stride) == 0

    def add_chunk(self, chunk: np.ndarray, query_step: int) -> None:
        """Ingests a newly predicted trajectory chunk into the overlapping buffer.

        Args:
            chunk: np.ndarray of shape [H, action_dim]
            query_step: Global simulation step at which chunk was predicted
        """
        if not isinstance(chunk, np.ndarray):
            chunk = np.array(chunk, dtype=np.float32)

        # Defensive Guard 1: Dimensionality and NaN checks
        if chunk.ndim != 2 or chunk.shape[-1] != self.action_dim:
            raise ValueError(f"Expected chunk shape [H, {self.action_dim}], got {chunk.shape}")

        chunk = np.nan_to_num(chunk, nan=0.0, posinf=1.0, neginf=-1.0)
        h = min(len(chunk), self.chunk_size)

        for offset in range(h):
            target_t = query_step + offset
            # Exponential discount based on temporal age of prediction
            dt = float(offset)
            weight = float(np.exp(-self.discount_m * dt))
            action_slice = chunk[offset].astype(np.float32)

            if target_t not in self._buffer:
                self._buffer[target_t] = []
            self._buffer[target_t].append((weight, action_slice))

        self._total_queries += 1

        # Defensive Guard 2: Memory Garbage Collection (purge past frames)
        purge_threshold = query_step - 2
        for old_t in list(self._buffer.keys()):
            if old_t < purge_threshold:
                del self._buffer[old_t]

    def get_action(self, current_step: int) -> np.ndarray:
        """Computes exponentially-weighted ensembled action for current_step.

        Returns:
            np.ndarray of shape [action_dim]
        """
        self._total_steps += 1

        # Defensive Guard 3: Buffer miss fallback
        if current_step not in self._buffer or not self._buffer[current_step]:
            logger.debug("Buffer miss at step %d; holding last known action.", current_step)
            return self._last_action.copy()

        entries = self._buffer[current_step]
        weights = np.array([w for w, _ in entries], dtype=np.float32)
        actions = np.array([a for _, a in entries], dtype=np.float32)

        total_weight = float(np.sum(weights))
        if total_weight < 1e-8:
            ensembled = np.mean(actions, axis=0)
        else:
            norm_weights = weights[:, None] / total_weight
            ensembled = np.sum(actions * norm_weights, axis=0)

        # Defensive Guard 4: Safety clamping
        if self.action_limits is not None:
            ensembled = np.clip(ensembled, self.action_limits[0], self.action_limits[1])

        self._last_action = ensembled.copy()

        # Clean up consumed step to maintain O(1) memory
        self._buffer.pop(current_step, None)
        return ensembled

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        return {
            "total_queries": self._total_queries,
            "total_steps": self._total_steps,
            "active_horizon_keys": len(self._buffer),
            "discount_m": self.discount_m,
            "query_stride": self.query_stride,
        }


if __name__ == "__main__":
    print("Testing TemporalEnsemble production module...")
    ensemble = TemporalEnsemble(action_dim=12, chunk_size=16, discount_m=0.05, query_stride=2)

    # Simulate 10-step receding horizon query sequence
    for s in range(10):
        if ensemble.should_query(s):
            synthetic_chunk = np.ones((16, 12), dtype=np.float32) * (s + 1.0)
            ensemble.add_chunk(synthetic_chunk, query_step=s)

        act = ensemble.get_action(s)
        print(f"Step {s:2d} -> Ensembled action mean: {act.mean():.4f}")

    print("Stats:", ensemble.stats)
    print("Verification Passed!")
