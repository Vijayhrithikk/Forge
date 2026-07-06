"""Execution Package — immutable bundle for remote execution."""
import json, hashlib, time
from pathlib import Path
from typing import Dict, Any

class ExecutionPackage:
    def __init__(self, training_plan: Dict, model_id: str, dataset_hash: str = "",
                 lora_config: Dict = None, hyperparams: Dict = None):
        self.data = {
            "schema_version": "1.0", "package_id": f"pkg_{int(time.time())}",
            "model": model_id, "dataset_hash": dataset_hash,
            "training_plan": training_plan,
            "lora": lora_config or {"rank": 8, "alpha": 16, "dropout": 0.05},
            "hyperparameters": hyperparams or {"epochs": 1, "batch_size": 1, "gradient_accumulation": 2,
                                                "learning_rate": 2e-4, "max_steps": 20, "precision": "bf16"},
            "created_at": time.time(),
        }
        raw = json.dumps(self.data, sort_keys=True)
        self._hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
        self.data["hash"] = self._hash

    @property
    def hash(self) -> str: return self._hash

    def to_dict(self) -> Dict: return dict(self.data)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f: json.dump(self.data, f, indent=2)
        return path
