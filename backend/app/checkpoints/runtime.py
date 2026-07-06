"""
Checkpoint Runtime — owns every checkpoint operation.
Trainer never writes directly to the filesystem.
"""

import json, hashlib, shutil, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from app.core import settings, get_logger

logger = get_logger("app.checkpoints.runtime")


class CheckpointRuntime:
    """Manages training checkpoint lifecycle: write, verify, register, retain, clean."""

    def __init__(self, checkpoints_dir: Path):
        self._dir = checkpoints_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, Dict] = {}

    def save(self, step: int, model: Any, optimizer: Any, scheduler: Any,
             trainer_state: Dict, runtime_id: str, session_id: str) -> Path:
        """Save a checkpoint. Trainer calls this through a callback."""
        t0 = time.time()
        ckpt_dir = self._dir / f"checkpoint-{step:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter and state (stubs — real saving requires transformers)
        manifest = {
            "checkpoint_id": f"ckpt-{step:06d}",
            "runtime_id": runtime_id, "session_id": session_id,
            "step": step, "timestamp": datetime.now(timezone.utc).isoformat(),
            "forge_version": settings.app_version, "schema_version": "1.0",
            "verification_status": "written",
        }
        with open(ckpt_dir / "checkpoint_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Register
        self._registry[f"checkpoint-{step:06d}"] = manifest
        self._save_registry()

        duration = round(time.time() - t0, 2)
        logger.info("checkpoint_saved", step=step, path=str(ckpt_dir), duration=duration)
        return ckpt_dir

    def verify(self, ckpt_dir: Path) -> Dict[str, str]:
        """Verify checkpoint integrity."""
        manifest_path = ckpt_dir / "checkpoint_manifest.json"
        if not manifest_path.exists():
            return {"status": "FAIL", "reason": "Manifest missing"}
        try:
            with open(manifest_path) as f:
                json.load(f)
        except Exception:
            return {"status": "FAIL", "reason": "Manifest corrupt"}
        return {"status": "PASS"}

    def list_checkpoints(self) -> list:
        """List all saved checkpoints."""
        result = []
        for d in sorted(self._dir.iterdir()):
            if d.is_dir() and d.name.startswith("checkpoint-"):
                mf = d / "checkpoint_manifest.json"
                if mf.exists():
                    with open(mf) as f:
                        result.append(json.load(f))
        return result

    def get_best(self) -> Optional[Path]:
        """Return the latest verified checkpoint path."""
        ckpts = self.list_checkpoints()
        if ckpts:
            return self._dir / ckpts[-1]["checkpoint_id"]
        return None

    def cleanup(self, keep_last: int = 3) -> int:
        """Remove old checkpoints, keeping the last N."""
        ckpts = self.list_checkpoints()
        removed = 0
        if len(ckpts) > keep_last:
            for old in ckpts[:-keep_last]:
                ckpt_dir = self._dir / old["checkpoint_id"]
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                    removed += 1
        if removed > 0:
            self._save_registry()
        logger.info("checkpoint_cleanup", removed=removed, kept=min(len(ckpts), keep_last))
        return removed

    def _save_registry(self):
        path = self._dir / "registry.json"
        with open(path, "w") as f:
            json.dump({"checkpoints": list(self._registry.values())}, f, indent=2)
