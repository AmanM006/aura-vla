import asyncio
from typing import Optional, Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import os

app = FastAPI(title="AI Observability Dashboard")

# SimState Schema
class SimState(BaseModel):
    timestamp: float
    phase: str
    sub_goals: Dict[str, bool]
    task_success: bool
    npu_ms: Optional[float]
    igpu_ms: Optional[float]
    cpu_ms: Optional[float]
    overhead_cam: Optional[str]
    left_wrist_cam: Optional[str]
    right_wrist_cam: Optional[str]
    transcript: str
    settled_text: str
    plan: List[str]
    plan_index: int
    objects: Dict[str, List[float]]
    anomaly_score: Optional[float] = 0.0
    anomaly_is_defect: Optional[bool] = False
    anomaly_hotspot: Optional[str] = "none"
    anomaly_device: Optional[str] = "OpenVINO (GPU)"
    policy_mode: Optional[str] = "diffusion"
    temporal_ensemble_active: Optional[bool] = True

# Global state
class GlobalState:
    def __init__(self):
        self.state = SimState(
            timestamp=0.0,
            phase="Idle",
            sub_goals={"drawer": False, "fork": False, "spoon": False, "plate": False, "mug": False},
            task_success=False,
            npu_ms=0.0,
            igpu_ms=0.0,
            cpu_ms=0.0,
            overhead_cam=None,
            left_wrist_cam=None,
            right_wrist_cam=None,
            transcript="",
            settled_text="",
            plan=[],
            plan_index=0,
            objects={}
        )
        self.instruction_queue = asyncio.Queue()
        self.clients: List[WebSocket] = []

global_state = GlobalState()

# Ensure static directory exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/status")
async def get_status():
    return global_state.state.dict()

class InstructionRequest(BaseModel):
    instruction: str

@app.post("/api/instruction")
async def post_instruction(req: InstructionRequest):
    await global_state.instruction_queue.put(req.instruction)
    return {"status": "queued", "instruction": req.instruction}

@app.websocket("/ws/state")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global_state.clients.append(websocket)
    try:
        while True:
            # Broadcast state every 100ms
            await websocket.send_text(global_state.state.model_dump_json())
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        global_state.clients.remove(websocket)
    except Exception as e:
        if websocket in global_state.clients:
            global_state.clients.remove(websocket)
