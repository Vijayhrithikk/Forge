"""
Validation Orchestrator — coordinates environment scanning, planning, preflight,
and authorization. The single authority for execution authorization.
"""

import json, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from app.core import settings, get_logger
from app.validation.environment.scanner import scanner
from app.validation.environment.planner import planner, risk_analyzer, recommendation_engine

logger = get_logger("app.validation.environment.orchestrator")


class ValidationStateMachine:
    """State machine for the validation pipeline."""
    STATES = ["INITIALIZING", "SCANNING", "ANALYZING", "PLANNING", "PREFLIGHT", "AUTHORIZING", "AUTHORIZED", "FAILED"]

    def __init__(self):
        self._state = "INITIALIZING"
        self._history: list = []

    @property
    def state(self) -> str: return self._state

    def transition(self, target: str) -> None:
        current_idx = self.STATES.index(self._state) if self._state in self.STATES else -1
        target_idx = self.STATES.index(target) if target in self.STATES else -1
        if target_idx < current_idx:
            raise ValueError(f"Illegal transition: {self._state} -> {target}")
        self._history.append({"from": self._state, "to": target, "at": datetime.now(timezone.utc).isoformat()})
        self._state = target


class ValidationOrchestrator:
    """Coordinates the complete validation pipeline: scan -> plan -> preflight -> authorize."""

    def __init__(self, workspace: Path = Path(".")):
        self._workspace = workspace
        self._sm = ValidationStateMachine()
        self._scan: Optional[Dict] = None
        self._plan: Optional[Dict] = None
        self._risks: Optional[Dict] = None
        self._recs: Optional[Dict] = None

    def run_full(self) -> Dict[str, Any]:
        """Run the complete validation pipeline."""
        t0 = time.time()

        self._sm.transition("SCANNING")
        self._scan = scanner.scan()
        scanner.save_report(self._scan, self._workspace / "environment_capability.json")

        self._sm.transition("ANALYZING")
        cap = {"hardware": self._scan.get("hardware", {}), "cuda": self._scan.get("cuda", {}),
               "dependencies": self._scan.get("dependencies", {}), "storage": self._scan.get("storage", {}),
               "network": self._scan.get("network", {})}

        self._sm.transition("PLANNING")
        self._plan = planner.plan(self._scan)
        self._risks = risk_analyzer.analyze(self._scan, self._plan)
        self._recs = recommendation_engine.recommend(self._scan)

        self._sm.transition("PREFLIGHT")
        preflight = self._run_preflight()

        self._sm.transition("AUTHORIZING")
        authorized = preflight["status"] == "PASS" and self._plan.get("can_execute", False)
        authorization = self._generate_authorization(authorized, preflight)

        self._sm.transition("AUTHORIZED" if authorized else "FAILED")

        result = {
            "orchestration": {
                "status": "AUTHORIZED" if authorized else "FAILED",
                "state": self._sm.state,
                "duration_seconds": round(time.time() - t0, 2),
            },
            "scan": self._scan, "plan": self._plan,
            "risks": self._risks, "recommendations": self._recs,
            "preflight": preflight, "authorization": authorization,
        }
        logger.info("validation_orchestration_complete", authorized=authorized)
        return result

    def _run_preflight(self) -> Dict[str, Any]:
        checks = []
        checks.append({"check": "workspace", "status": "PASS" if self._workspace.is_dir() else "FAIL",
                        "detail": "Workspace exists" if self._workspace.is_dir() else "Workspace missing"})
        checks.append({"check": "disk_space", "status": "PASS" if self._scan.get("storage", {}).get("workspace", {}).get("free_gb", 0) >= 10 else "FAIL",
                        "detail": f"Disk: {self._scan.get('storage', {}).get('workspace', {}).get('free_gb', 0)}GB free"})
        checks.append({"check": "cuda", "status": "PASS" if self._scan.get("cuda", {}).get("cuda_available") else "FAIL",
                        "detail": "CUDA available" if self._scan.get("cuda", {}).get("cuda_available") else "No CUDA GPU"})
        dep_fails = sum(1 for d in self._scan.get("dependencies", {}).values() if d.get("status") == "FAIL" and d.get("required_pkg"))
        checks.append({"check": "dependencies", "status": "PASS" if dep_fails == 0 else "FAIL",
                        "detail": f"{dep_fails} required packages missing"})
        status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
        return {"status": status, "checks": checks}

    def _generate_authorization(self, authorized: bool, preflight: Dict) -> Dict[str, Any]:
        auth = {
            "authorization_id": f"auth_{int(time.time())}",
            "authorized": authorized,
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": datetime.now(timezone.utc).isoformat() if authorized else None,
            "forge_version": settings.app_version, "schema_version": "1.0",
            "model": self._plan.get("selected_model") if self._plan else None,
            "preflight_status": preflight["status"],
            "restrictions": [],
        }
        if not authorized:
            auth["restrictions"] = [c["detail"] for c in preflight.get("checks", []) if c["status"] == "FAIL"]
        path = self._workspace / "execution_authorization.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(auth, f, indent=2, ensure_ascii=False)
        return auth
