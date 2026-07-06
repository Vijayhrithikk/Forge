"""Execution Planner — matches hardware to models and generates safe validation plans."""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
from app.core import settings, get_logger
from app.engines.training.registry import model_registry

logger = get_logger("app.validation.environment.planner")

MODEL_PREFERENCE = ["qwen2.5-1.5b-instruct", "tinyllama-1.1b-chat",
                     "phi-4-mini-instruct", "gemma-3-4b-it", "llama-3.2-3b-instruct"]


class ExecutionPlanner:
    """Generates a safe execution validation plan based on environment capabilities."""

    def plan(self, scan: Dict[str, Any]) -> Dict[str, Any]:
        hw = scan.get("hardware", {})
        cu = scan.get("cuda", {})
        deps = scan.get("dependencies", {})
        disk = scan.get("storage", {})

        has_cuda = cu.get("cuda_available", False)
        vram_gb = hw.get("gpus", [{}])[0].get("vram_gb", 0) if hw.get("gpus") else 0
        ram_gb = hw.get("ram_total_gb", 0)
        free_disk_gb = disk.get("workspace", {}).get("free_gb", 0)

        # Select best model for validation
        selected_model = self._select_model(vram_gb, deps)
        model = model_registry.get_model(selected_model) if selected_model else None

        precision = "bf16" if cu.get("checks", {}).get("bf16", {}).get("status") == "PASS" else "fp16"
        batch_size = 1 if vram_gb < 12 else 2

        plan = {
            "schema_version": "1.0", "forge_version": settings.app_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selected_model": selected_model,
            "model_name": model.name if model else "none",
            "precision": precision, "batch_size": batch_size,
            "gradient_accumulation": 2,
            "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
            "epochs": 1, "max_steps": 20,
            "estimated_vram_gb": round(vram_gb * 0.8, 1) if vram_gb > 0 else 0,
            "estimated_ram_gb": round(ram_gb * 0.5, 1),
            "estimated_runtime_minutes": 10,
            "expected_adapter_size_mb": 5.0,
            "risk_level": "LOW" if has_cuda and vram_gb >= 8 else "HIGH",
            "can_execute": has_cuda and selected_model is not None and free_disk_gb >= 10,
            "limitations": [],
        }

        if not has_cuda: plan["limitations"].append("No CUDA GPU detected")
        if vram_gb < 6: plan["limitations"].append(f"Low VRAM ({vram_gb}GB)")
        if free_disk_gb < 10: plan["limitations"].append(f"Low disk space ({free_disk_gb}GB)")

        logger.info("execution_plan_generated", model=selected_model, can_execute=plan["can_execute"])
        return plan

    def _select_model(self, vram_gb: float, deps: Dict) -> str:
        for model_id in MODEL_PREFERENCE:
            try:
                model = model_registry.get_model(model_id)
                if vram_gb >= model.recommended_vram_gb:
                    return model_id
            except KeyError:
                continue
        return MODEL_PREFERENCE[0]  # Fallback

    def save_plan(self, plan: Dict, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        return path


class RiskAnalyzer:
    """Analyzes execution risks across GPU, memory, disk, dependencies, and network."""

    def analyze(self, scan: Dict, plan: Dict) -> Dict[str, Any]:
        cu = scan.get("cuda", {})
        disk = scan.get("storage", {})
        deps = scan.get("dependencies", {})
        net = scan.get("network", {})

        risks = {}
        risks["gpu"] = self._level("HIGH" if not cu.get("cuda_available") else "LOW",
                                    "No CUDA GPU" if not cu.get("cuda_available") else "GPU available")
        risks["memory"] = self._level("LOW" if plan.get("estimated_vram_gb", 0) > 2 else "MEDIUM",
                                       f"VRAM: {plan.get('estimated_vram_gb', 0)}GB")
        risks["disk"] = self._level("LOW" if disk.get("workspace", {}).get("free_gb", 0) >= 10 else "HIGH",
                                     f"Free: {disk.get('workspace', {}).get('free_gb', 0)}GB")
        dep_fails = sum(1 for d in deps.values() if d.get("status") == "FAIL" and d.get("required_pkg"))
        risks["dependencies"] = self._level("HIGH" if dep_fails > 0 else "LOW",
                                              f"{dep_fails} required packages missing")
        risks["network"] = self._level("LOW" if net.get("internet", {}).get("status") == "PASS" else "MEDIUM",
                                        "Internet: " + net.get("internet", {}).get("status", "unknown"))
        overall = "LOW" if all(r["level"] == "LOW" for r in risks.values()) else \
                  "CRITICAL" if any(r["level"] == "HIGH" for r in risks.values()) else "MEDIUM"

        return {"risks": risks, "overall_level": overall,
                "generated_at": datetime.now(timezone.utc).isoformat()}

    def _level(self, level: str, detail: str) -> Dict: return {"level": level, "detail": detail}


class RecommendationEngine:
    """Generates actionable recommendations from scan results."""

    def recommend(self, scan: Dict) -> Dict[str, Any]:
        recs = []
        cu = scan.get("cuda", {})
        deps = scan.get("dependencies", {})
        disk = scan.get("storage", {})

        if not cu.get("cuda_available"):
            recs.append({"priority": "CRITICAL", "action": "Install CUDA-capable GPU with 8GB+ VRAM"})
        for name, d in deps.items():
            if d.get("status") == "FAIL" and d.get("required_pkg"):
                recs.append({"priority": "HIGH", "action": f"Install {name}>={d['required']}"})
            elif d.get("status") == "WARNING":
                recs.append({"priority": "MEDIUM", "action": f"Consider installing {name}>={d['required']}"})
        for loc, info in disk.items():
            if isinstance(info, dict) and info.get("status") == "FAIL":
                recs.append({"priority": "HIGH", "action": f"Free disk space on {loc} ({info.get('free_gb', 0)}GB free, need 10GB)"})

        return {"recommendations": recs,
                "generated_at": datetime.now(timezone.utc).isoformat()}


planner = ExecutionPlanner()
risk_analyzer = RiskAnalyzer()
recommendation_engine = RecommendationEngine()
