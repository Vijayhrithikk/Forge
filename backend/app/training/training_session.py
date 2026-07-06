"""
Training Session — the complete training execution context.

Tracks session ID, all component hashes, current state, metrics,
and produces training_session.json for reproducibility and recovery.
"""

import json, hashlib, uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from app.core import settings, get_logger

logger = get_logger("app.training.training_session")


class TrainingSession:
    """A complete training session — the unit of training execution."""

    def __init__(self, runtime_id: str, project_id: str):
        self.session_id = f"train_{uuid.uuid4().hex[:12]}"
        self.runtime_id = runtime_id
        self.project_id = project_id
        self.state = "CREATED"
        self._hashes: Dict[str, str] = {}
        self._metrics: list = []
        self._created_at = datetime.now(timezone.utc).isoformat()

    def set_hash(self, component: str, hash_value: str) -> None:
        self._hashes[component] = hash_value

    def update_state(self, state: str) -> None:
        self.state = state

    def add_metrics(self, metrics: list) -> None:
        self._metrics = metrics

    def generate(self, execution_graph_hash: str = "") -> Dict[str, Any]:
        session_hash_input = json.dumps({
            "session_id": self.session_id, "runtime_id": self.runtime_id,
            **self._hashes, "state": self.state,
        }, sort_keys=True)
        session_hash = hashlib.sha256(session_hash_input.encode()).hexdigest()[:16]

        return {
            "schema_version": "1.0", "forge_version": settings.app_version,
            "session_id": self.session_id, "runtime_id": self.runtime_id,
            "project_id": self.project_id, "state": self.state,
            "session_hash": session_hash, "execution_graph_hash": execution_graph_hash,
            "component_hashes": self._hashes,
            "created_at": self._created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, directory: Path, graph_hash: str = "") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "training_session.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.generate(graph_hash), f, indent=2, ensure_ascii=False)
        logger.info("training_session_saved", session_id=self.session_id)
        return path
