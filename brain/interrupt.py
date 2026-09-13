import threading
from typing import Optional, Callable, Any

class InterruptHandler:
    """Thread-safe interrupt + replan for the control loop."""
    def __init__(self, planner: Any, controller: Any):
        self.planner = planner
        self.controller = controller
        self._lock = threading.Lock()
        self._interrupted = False
        self._new_instruction: Optional[str] = None

    def request_interrupt(self, new_instruction: str):
        """Thread-safe: signals the control loop to halt after current step"""
        with self._lock:
            self._interrupted = True
            self._new_instruction = new_instruction

    def check_and_handle(self, model: Any, data: Any, scene_state: dict) -> Optional[list]:
        """Call this from the control loop. Returns new plan if interrupted, else None."""
        with self._lock:
            if not self._interrupted:
                return None
            
            instruction = self._new_instruction
            self._interrupted = False
            self._new_instruction = None
            
        # Replan based on new instruction and current state
        if hasattr(self.planner, 'replan'):
            new_plan = self.planner.replan(instruction, scene_state)
            return new_plan
        return [f"Handle interrupt: {instruction}"]

    def is_interrupted(self) -> bool:
        with self._lock:
            return self._interrupted
