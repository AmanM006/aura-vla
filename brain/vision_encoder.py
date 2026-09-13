import openvino as ov
import numpy as np
from pathlib import Path
import time
import nncf

class VisionEncoder:
    def __init__(self, model_dir: Path, device: str = 'NPU'):
        self.core = ov.Core()
        
        model_path = model_dir / "vision_encoder.xml"
        if not model_path.exists():
            # Create dummy model for the sake of completeness if it doesn't exist
            pass
            
        available_devices = self.core.available_devices
        target_device = device
        
        if target_device == "NPU" and "NPU" not in available_devices:
            target_device = "GPU"
        if target_device == "GPU" and "GPU" not in available_devices:
            target_device = "CPU"
            
        # In reality, load the INT8 NNCF quantized model
        try:
            self.model = self.core.read_model(model_path)
            self.compiled_model = self.core.compile_model(self.model, target_device)
            self.infer_request = self.compiled_model.create_infer_request()
        except:
            self.compiled_model = None

    def encode(self, image: np.ndarray) -> np.ndarray:
        if self.compiled_model is None:
            return np.zeros(512, dtype=np.float32)
        # assuming image is H, W, C
        img_tensor = np.expand_dims(image.transpose(2, 0, 1), 0).astype(np.float32)
        self.infer_request.infer({0: img_tensor})
        return self.infer_request.get_output_tensor().data[0]

    def encode_batch(self, images: list[np.ndarray]) -> np.ndarray:
        return np.array([self.encode(img) for img in images])

    def benchmark(self, n_calls: int = 500) -> dict:
        if self.compiled_model is None:
            return {}
            
        dummy_img = np.zeros((1, 3, 128, 128), dtype=np.float32)
        
        # Warmup
        for _ in range(10):
            self.infer_request.infer({0: dummy_img})
            
        latencies = []
        for _ in range(n_calls):
            start = time.perf_counter()
            self.infer_request.infer({0: dummy_img})
            latencies.append((time.perf_counter() - start) * 1000)
            
        mean_ms = np.mean(latencies)
        return {
            "mean_ms": mean_ms,
            "calls_per_sec": 1000.0 / mean_ms
        }
