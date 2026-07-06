"""
Runtime Events — strongly typed, append-only event stream.

Every significant Runtime action emits an event. Events are stored
as JSON Lines (.jsonl) for easy streaming and post-hoc analysis.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional

from app.core import get_logger

logger = get_logger("app.runtime.events")


class EventSeverity(str, Enum):
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


class RuntimeEventType(str, Enum):
    """Every possible Runtime event type."""
    # Lifecycle
    RUNTIME_CREATED = "RUNTIME_CREATED"
    RUNTIME_READY = "RUNTIME_READY"
    RUNTIME_FAILED = "RUNTIME_FAILED"
    RUNTIME_CANCELLED = "RUNTIME_CANCELLED"

    # Validation
    VALIDATION_STARTED = "VALIDATION_STARTED"
    VALIDATION_COMPLETED = "VALIDATION_COMPLETED"

    # Environment
    ENVIRONMENT_VALIDATION_STARTED = "ENVIRONMENT_VALIDATION_STARTED"
    ENVIRONMENT_VALIDATION_COMPLETED = "ENVIRONMENT_VALIDATION_COMPLETED"

    # GPU
    GPU_DISCOVERY_STARTED = "GPU_DISCOVERY_STARTED"
    GPU_DISCOVERY_COMPLETED = "GPU_DISCOVERY_COMPLETED"
    GPU_VALIDATION_STARTED = "GPU_VALIDATION_STARTED"
    GPU_VALIDATION_COMPLETED = "GPU_VALIDATION_COMPLETED"

    # Workspace
    WORKSPACE_PREPARATION_STARTED = "WORKSPACE_PREPARATION_STARTED"
    WORKSPACE_PREPARATION_COMPLETED = "WORKSPACE_PREPARATION_COMPLETED"

    # Lock
    LOCK_ACQUIRED = "LOCK_ACQUIRED"
    LOCK_RELEASED = "LOCK_RELEASED"

    # Recovery
    RECOVERY_ATTEMPTED = "RECOVERY_ATTEMPTED"
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"
    RECOVERY_FAILED = "RECOVERY_FAILED"


class RuntimeEvent:
    """A single Runtime event with full metadata."""

    def __init__(
        self,
        event_type: RuntimeEventType,
        runtime_id: str,
        severity: EventSeverity = EventSeverity.INFO,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.event_id = f"evt_{uuid.uuid4().hex[:12]}"
        self.event_type = event_type
        self.runtime_id = runtime_id
        self.severity = severity
        self.message = message
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "runtime_id": self.runtime_id,
            "severity": self.severity.value,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class EventStore:
    """Append-only event persistence backed by JSON Lines.

    Events are written one per line as JSON objects. This enables
    streaming consumption by the frontend and post-hoc analysis.
    """

    def __init__(self, events_dir: Path, runtime_id: str):
        self._path = events_dir / "events.jsonl"
        self._runtime_id = runtime_id
        events_dir.mkdir(parents=True, exist_ok=True)

    def append(self, event: RuntimeEvent) -> None:
        """Append an event to the event stream."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def read_all(self) -> list[Dict[str, Any]]:
        """Read all events from the stream."""
        if not self._path.exists():
            return []
        events = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def read_recent(self, limit: int = 50) -> list[Dict[str, Any]]:
        """Read the most recent events (up to limit)."""
        all_events = self.read_all()
        return all_events[-limit:] if len(all_events) > limit else all_events

    def clear(self) -> None:
        """Clear all events (for testing/reset)."""
        if self._path.exists():
            self._path.unlink()
