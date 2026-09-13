import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from policy.openvino_runtime import ACTOpenVINORuntime
from brain.planner import VLMPlanner
from brain.vision_encoder import VisionEncoder

def required_hardware_verdict() -> str:
    try:
        import cpuinfo
        cpu_name = cpuinfo.get_cpu_info().get('brand_raw', '')
    except ImportError:
        cpu_name = "Unknown"
        
    core_ultra_patterns = ['Core Ultra', 'Core(TM) Ultra']
    if any(p in cpu_name for p in core_ultra_patterns):
        return 'MEASURED_ON_REQUIRED_HARDWARE'
    return f'NOT_THE_REQUIRED_MEASUREMENT (host: {cpu_name})'

def run_benchmark(args):
    console = Console()
    console.print("[bold green]Starting OpenVINO Heterogeneous Pipeline Benchmark[/bold green]")
    
    devices = ['CPU', 'GPU', 'NPU']
    precisions = ['fp32', 'fp16', 'int8']
    
    evidence_dir = Path("evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    verdict = required_hardware_verdict()
    console.print(f"Hardware Verdict: [bold blue]{verdict}[/bold blue]")
    
    instructions = ["set the table", "put the red thing top left of the plate", "grab the fork"]
    
    for device in devices:
        for precision in precisions:
            console.print(f"\n[bold]Benchmarking on {device} with {precision}[/bold]")
            
            try:
                act_rt = ACTOpenVINORuntime(Path("models/act_openvino"), device=device, precision=precision)
                act_metrics = act_rt.benchmark()
            except Exception as e:
                act_metrics = {"error": str(e)}
                
            try:
                vision = VisionEncoder(Path("models/vision"), device=device)
                vision_metrics = vision.benchmark()
            except Exception as e:
                vision_metrics = {"error": str(e)}
                
            try:
                planner = VLMPlanner(device=device)
                planner_metrics = planner.benchmark(instructions)
            except Exception as e:
                planner_metrics = {"error": str(e)}
                
            results = {
                "hardware_verdict": verdict,
                "device": device,
                "precision": precision,
                "act_policy": act_metrics,
                "vision_encoder": vision_metrics,
                "vlm_planner": planner_metrics
            }
            
            with open(evidence_dir / f"openvino_bench_{device}_{precision}.json", "w") as f:
                json.dump(results, f, indent=4)
                
            table = Table(title=f"Results: {device} ({precision})")
            table.add_column("Component", style="cyan")
            table.add_column("Mean Latency", justify="right")
            table.add_column("Throughput", justify="right")
            
            if "mean_ms" in act_metrics:
                table.add_row("ACT Policy", f"{act_metrics['mean_ms']:.2f} ms", f"{act_metrics['calls_per_sec']:.2f} iter/s")
            if "mean_ms" in vision_metrics:
                table.add_row("Vision Encoder", f"{vision_metrics['mean_ms']:.2f} ms", f"{vision_metrics.get('calls_per_sec', 0):.2f} img/s")
            if "mean_s" in planner_metrics:
                table.add_row("VLM Planner", f"{planner_metrics['mean_s']:.2f} s", f"{planner_metrics.get('tokens_per_s', 0):.2f} tok/s")
                
            console.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    run_benchmark(args)
