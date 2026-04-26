import json
import os
import subprocess
import importlib
import asyncio
import inspect
from datetime import datetime
from typing import List, Dict, Any, Optional
from ..core.utils import write_text

class Runner:
    def __init__(self, job_dir: str):
        self.job_dir = job_dir
        self.steps_dir = os.path.join(job_dir, "steps")
        os.makedirs(self.steps_dir, exist_ok=True)
        # Ensure steps is a Python package
        init_path = os.path.join(self.steps_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("")

    def save_step_module(self, step_id: str, code: str):
        path = os.path.join(self.steps_dir, f"step_{step_id}.py")
        write_text(path, code)
        return path

    async def run_in_process(self, tab: Any, step_ids: List[str], shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run steps in the current process. 
        tab should be a PageAdapter implementation.
        """
        errors = {}
        for sid in step_ids:
            print(f"[Runner] Starting step: {sid}")
            try:
                # Add job_dir to sys.path so we can import the generated steps
                import sys
                if self.job_dir not in sys.path:
                    sys.path.insert(0, self.job_dir)
                
                # Import the step module
                module_name = f"steps.step_{sid}"
                # Force reload if it was already imported
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                
                mod = importlib.import_module(module_name)
                if not hasattr(mod, "run_step"):
                    raise RuntimeError(f"Step {sid} missing run_step function")
                
                result = mod.run_step(tab, shared)
                if inspect.iscoroutine(result):
                    await result
                print(f"[Runner] Step {sid} completed successfully")
            except Exception as e:
                print(f"[Runner] Step {sid} failed: {e}")
                errors[sid] = str(e)
                shared.setdefault("errors", {})[sid] = str(e)
                # Decision: stop or continue? 
                # Standard autonomous agent usually stops on fatal errors.
                if sid != "handle_blockers":
                    break
        
        return {"ok": len(errors) == 0, "errors": errors}

    # We could also keep the write_runner_py approach for isolation
    def generate_standalone_runner(self, step_ids: List[str], url: str, intent: Dict[str, bool]) -> str:
        # Implementation similar to pydoll_agent/runner_gen.py
        pass
