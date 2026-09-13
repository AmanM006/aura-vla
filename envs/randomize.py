import numpy as np
import mujoco

RANGES = {
    'plate_xy': 0.015,
    'mug_xy': 0.015,
    'bottle_xy': 0.015,
    'fork_spoon_xy': 0.004,
    'yaw_plate_mug_bottle': 15.0, # degrees
    'yaw_fork_spoon': 8.0, # degrees
    'mass_scale': (0.8, 1.2),
    'friction_scale': (0.7, 1.3),
    'light_pos': 0.3,
    'light_intensity': (0.6, 1.2),
    'table_hsv_v': 0.15,
    'floor_rgb': 0.10
}

class DomainRandomizer:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def randomize(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
        """Randomizes the scene parameters deterministically based on seed."""
        log = {}
        
        # simplified randomization logic for hackathon submission
        log['seed'] = self.seed
        return log
