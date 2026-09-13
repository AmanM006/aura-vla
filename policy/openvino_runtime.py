import openvino as ov
import numpy as np
from pathlib import Path
import time

class ACTOpenVINORuntime:
    def __init__(self, model_dir: Path, device: str = 'CPU', precision: str = 'fp32'):
        self.core = ov.Core()
        model_path = model_dir / precision / "act.xml"
        
        available_devices = self.core.available_devices
        target_device = device
        
        # Fallback chain: NPU -> GPU -> CPU
        if target_device == "NPU" and "NPU" not in available_devices:
            target_device = "GPU"
        if target_device == "GPU" and "GPU" not in available_devices:
            target_device = "CPU"
            
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, target_device)
        self.infer_request = self.compiled_model.create_infer_request()
        
    def predict(self, obs_state: np.ndarray, images: list[np.ndarray], task_token: int) -> np.ndarray:
        # Preprocess images (assuming they are BGR/RGB np.ndarray H, W, C)
        # Convert to 1, 3, 128, 128
        processed_imgs = []
        for img in images:
            # Assuming already 128x128x3
            img_tensor = np.expand_dims(img.transpose(2, 0, 1), 0).astype(np.float32)
            processed_imgs.append(img_tensor)
            
        obs = np.expand_dims(obs_state, 0).astype(np.float32)
        task = np.zeros((1, 10), dtype=np.float32)
        task[0, task_token] = 1.0
        
        inputs = [obs, processed_imgs[0], processed_imgs[1], processed_imgs[2], task]
        
        self.infer_request.infer(inputs)
        actions = self.infer_request.get_output_tensor().data
        # Return first 25 actions for the chunk
        return actions[0, :25]
        
    def benchmark(self, n_calls: int = 200, n_warmup: int = 10) -> dict:
        dummy_inputs = [
            np.zeros((1, 12), dtype=np.float32),
            np.zeros((1, 3, 128, 128), dtype=np.float32),
            np.zeros((1, 3, 128, 128), dtype=np.float32),
            np.zeros((1, 3, 128, 128), dtype=np.float32),
            np.zeros((1, 10), dtype=np.float32)
        ]
        
        for _ in range(n_warmup):
            self.infer_request.infer(dummy_inputs)
            
        latencies = []
        for _ in range(n_calls):
            start = time.perf_counter()
            self.infer_request.infer(dummy_inputs)
            latencies.append((time.perf_counter() - start) * 1000)
            
        latencies = np.array(latencies)
        mean_ms = np.mean(latencies)
        calls_per_sec = 1000.0 / mean_ms
        exec_devices = self.compiled_model.get_property("EXECUTION_DEVICES")
        
        return {
            "mean_ms": mean_ms,
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "calls_per_sec": calls_per_sec,
            "execution_precision": exec_devices
        }
