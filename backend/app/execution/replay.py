"""Execution Replay — reconstructs execution context from session artifacts."""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from app.core import settings

def reconstruct_from_session(session_path: Path) -> Dict[str, Any]:
    if not session_path.exists():
        return {"status": "FAIL", "reason": "Session file not found"}
    try:
        with open(session_path) as f: session = json.load(f)
    except Exception as e:
        return {"status": "FAIL", "reason": f"Session corrupt: {e}"}

    replay = {
        "schema_version": "1.0", "forge_version": settings.app_version,
        "replayed_at": datetime.now(timezone.utc).isoformat(),
        "original_session_id": session.get("session_id"),
        "execution_target": session.get("execution_target"),
        "model": session.get("model"),
        "state": session.get("state"),
        "stages": session.get("stages", {}),
        "reconstruction_possible": True,
    }
    path = session_path.parent / "execution_replay.json"
    with open(path, "w") as f: json.dump(replay, f, indent=2)
    return replay
