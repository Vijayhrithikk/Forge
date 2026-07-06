"""
Execution Graph — serializable, deterministic training pipeline representation.

Documents every node in the training execution: PreparedModel,
DatasetRuntime, PEFTRuntime, TrainerBuilder, TrainingController.
Generates execution_graph.json for reproducibility.
"""

import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
from app.core import settings, get_logger

logger = get_logger("app.training.execution_graph")


class ExecutionGraph:
    """Deterministic graph of training execution nodes."""

    NODE_ORDER = [
        "prepared_model", "dataset_runtime", "tokenization_runtime",
        "peft_runtime", "trainer_builder", "training_controller",
    ]

    def __init__(self, runtime_id: str):
        self._runtime_id = runtime_id
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: List[tuple] = []

    def add_node(self, name: str, status: str = "PENDING", metadata: Dict = None) -> None:
        self._nodes[name] = {"status": status, "metadata": metadata or {}}

    def add_edge(self, from_node: str, to_node: str) -> None:
        self._edges.append((from_node, to_node))

    def build_default(self) -> None:
        """Build the standard training execution graph."""
        for i, name in enumerate(self.NODE_ORDER):
            self.add_node(name)
            if i > 0:
                self.add_edge(self.NODE_ORDER[i - 1], name)

    def validate(self) -> Dict[str, Any]:
        """Validate graph completeness."""
        ready = all(n["status"] == "READY" for n in self._nodes.values())
        return {
            "all_ready": ready,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "nodes": list(self._nodes.keys()),
        }

    def compute_hash(self) -> str:
        data = json.dumps({
            "nodes": {k: v["status"] for k, v in sorted(self._nodes.items())},
            "edges": sorted(self._edges),
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def generate(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0", "forge_version": settings.app_version,
            "runtime_id": self._runtime_id,
            "graph_hash": self.compute_hash(),
            "nodes": self._nodes, "edges": [[a, b] for a, b in self._edges],
            "node_order": self.NODE_ORDER,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "execution_graph.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.generate(), f, indent=2, ensure_ascii=False)
        return path
