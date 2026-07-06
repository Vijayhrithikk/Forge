"""Provider State Machine — explicit lifecycle states."""
from datetime import datetime, timezone
from typing import Dict, Any, List

VALID_TRANSITIONS = {
    "IDLE": ["PREPARING"], "PREPARING": ["UPLOADING", "FAILED"],
    "UPLOADING": ["EXECUTING", "FAILED"], "EXECUTING": ["STREAMING", "DOWNLOADING", "FAILED"],
    "STREAMING": ["EXECUTING", "DOWNLOADING", "FAILED"], "DOWNLOADING": ["CLEANING", "FAILED"],
    "CLEANING": ["COMPLETED", "FAILED"], "COMPLETED": [], "FAILED": [],
}

class ProviderStateMachine:
    def __init__(self):
        self._state = "IDLE"; self._history: List[Dict] = []
    @property
    def state(self) -> str: return self._state
    def transition(self, target: str) -> None:
        if target not in VALID_TRANSITIONS.get(self._state, []):
            raise ValueError(f"Illegal transition: {self._state} -> {target}")
        self._history.append({"from": self._state, "to": target, "at": datetime.now(timezone.utc).isoformat()})
        self._state = target
