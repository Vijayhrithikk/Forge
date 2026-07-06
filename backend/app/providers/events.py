"""Event Translation — maps provider events to Forge Runtime events."""
from typing import Dict, Callable, Optional

FORGE_EVENT_MAP = {
    "job_started": "EXECUTION_STARTED", "job_completed": "TRAINING_COMPLETED",
    "checkpoint_created": "CHECKPOINT_CREATED", "artifact_ready": "ARTIFACT_READY",
    "job_failed": "EXECUTION_FAILED", "stream_ready": "STREAM_CONNECTED",
}

class EventTranslator:
    def __init__(self, on_forge_event: Optional[Callable] = None):
        self._on_event = on_forge_event or (lambda *a, **kw: None)
    def translate(self, provider_event: str, metadata: Dict = None) -> str:
        forge_event = FORGE_EVENT_MAP.get(provider_event, f"PROVIDER_{provider_event.upper()}")
        self._on_event(forge_event, metadata or {})
        return forge_event
