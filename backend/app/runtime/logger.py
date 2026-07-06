"""
Runtime Logger — structured, JSON Lines logging for execution.

Every log entry contains: timestamp, runtime_id, project_id, stage,
level, component, message, and metadata. Never use print().
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.core import get_logger as get_structlog_logger

# Underlying structlog logger for console output
_base_logger = get_structlog_logger("app.runtime.logger")


class RuntimeLogger:
    """Owns all execution logs for a Runtime instance.

    Writes structured JSON Lines to disk AND emits through structlog
    for console/aggregation visibility.
    """

    def __init__(self, logs_dir: Path, runtime_id: str, project_id: str):
        self._path = logs_dir / "logs.jsonl"
        self._runtime_id = runtime_id
        self._project_id = project_id
        logs_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        level: str,
        stage: str,
        component: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a structured log entry.

        Args:
            level: INFO, SUCCESS, WARNING, ERROR
            stage: Current Runtime stage (e.g., VALIDATING, PREPARING)
            component: Module name (e.g., gpu, environment, workspace)
            message: Human-readable description
            metadata: Additional structured data
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime_id": self._runtime_id,
            "project_id": self._project_id,
            "stage": stage,
            "level": level,
            "component": component,
            "message": message,
            "metadata": metadata or {},
        }

        # Disk persistence
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Console visibility
        log_method = getattr(_base_logger, level.lower(), _base_logger.info)
        log_method(message, **{k: v for k, v in entry.items() if k != "message"})

    def info(self, stage: str, component: str, message: str, **meta) -> None:
        self.log("INFO", stage, component, message, meta if meta else None)

    def success(self, stage: str, component: str, message: str, **meta) -> None:
        self.log("SUCCESS", stage, component, message, meta if meta else None)

    def warning(self, stage: str, component: str, message: str, **meta) -> None:
        self.log("WARNING", stage, component, message, meta if meta else None)

    def error(self, stage: str, component: str, message: str, **meta) -> None:
        self.log("ERROR", stage, component, message, meta if meta else None)

    def read_all(self) -> list[dict]:
        """Read all log entries from disk."""
        if not self._path.exists():
            return []
        entries = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
