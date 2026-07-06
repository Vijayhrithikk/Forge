"""Adapter Validator — validates LoRA adapter files."""
import json, hashlib
from pathlib import Path
from typing import Dict, Any
from app.core import get_logger

logger = get_logger("app.validation.adapter_validator")


class AdapterValidator:
    def validate(self, output_dir: Path) -> Dict[str, Any]:
        safetensors = output_dir / "adapter_model.safetensors"
        config = output_dir / "adapter_config.json"
        results = {}
        if safetensors.exists():
            sha = hashlib.sha256(safetensors.read_bytes()).hexdigest()
            results["adapter_weights"] = {"status": "PASS", "hash": sha[:16], "size": safetensors.stat().st_size}
        else:
            results["adapter_weights"] = {"status": "MISSING"}
        if config.exists():
            with open(config) as f:
                data = json.load(f)
            results["adapter_config"] = {"status": "PASS", "rank": data.get("r"), "alpha": data.get("lora_alpha")}
        else:
            results["adapter_config"] = {"status": "MISSING"}
        status = "PASS" if all(r["status"] == "PASS" for r in results.values()) else "WARNING"
        return {"status": status, "results": results}
