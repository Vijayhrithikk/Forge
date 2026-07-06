"""Execution Certificate — truthful validation result."""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from app.core import settings

def generate_certificate(session: Dict, adapter_ok: bool, inference_ok: bool,
                         training_attempted: bool, training_succeeded: bool,
                         skip_reason: str = "") -> Dict[str, Any]:
    if not training_attempted:
        overall = "SKIPPED"
    elif training_succeeded and adapter_ok and inference_ok:
        overall = "PASS"
    else:
        overall = "FAIL"

    cert = {
        "schema_version": "1.0", "forge_version": settings.app_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_target": session.get("execution_target"),
        "model": session.get("model"),
        "training": {"attempted": training_attempted, "succeeded": training_succeeded},
        "adapter": {"validated": adapter_ok},
        "inference": {"validated": inference_ok},
        "overall_result": overall,
    }
    if skip_reason: cert["skip_reason"] = skip_reason
    return cert
