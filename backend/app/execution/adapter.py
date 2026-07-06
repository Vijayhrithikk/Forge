"""Execution Adapter — common interface for target environments."""
from pathlib import Path
from typing import Dict, Any
from app.execution.targets import detect_target, ExecutionTarget

class ExecutionAdapter:
    """Prepares the execution environment for the detected target."""
    def __init__(self):
        self._target = detect_target()

    @property
    def target(self) -> ExecutionTarget: return self._target

    def prepare(self) -> Dict[str, Any]:
        workspace = Path(self._target.workspace_prefix)
        cache = Path(self._target.cache_prefix)
        output = Path(self._target.output_prefix)
        for d in [workspace, cache, output]:
            d.mkdir(parents=True, exist_ok=True)
        return {"target": self._target.name, "workspace": str(workspace),
                "cache": str(cache), "output": str(output), "ready": True}
