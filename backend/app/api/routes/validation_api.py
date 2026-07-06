"""Validation API — environment scanning, capability assessment, execution authorization."""
from pathlib import Path
from fastapi import APIRouter, HTTPException

from app.core import get_logger
from app.schemas.responses import SuccessResponse
from app.validation.environment.scanner import scanner
from app.validation.environment.planner import planner, risk_analyzer, recommendation_engine
from app.validation.environment.orchestrator import ValidationOrchestrator

router = APIRouter(prefix="/validation", tags=["validation"])
logger = get_logger("app.api.validation")


@router.get("/environment", response_model=SuccessResponse)
async def get_environment():
    """Run and return a complete environment capability scan."""
    report = scanner.scan()
    scanner.save_report(report, Path("data") / "environment_capability.json")
    return SuccessResponse(message=f"Environment: {report['overall_status']}", data=report)


@router.get("/capabilities", response_model=SuccessResponse)
async def get_capabilities():
    """Get capability analysis from the latest scan."""
    scan = scanner.scan()
    plan = planner.plan(scan)
    return SuccessResponse(message=f"Capability: {'ready' if plan['can_execute'] else 'limited'}", data=plan)


@router.get("/recommendations", response_model=SuccessResponse)
async def get_recommendations():
    """Get environment improvement recommendations."""
    scan = scanner.scan()
    recs = recommendation_engine.recommend(scan)
    return SuccessResponse(message=f"{len(recs['recommendations'])} recommendations", data=recs)


@router.get("/execution-plan", response_model=SuccessResponse)
async def get_execution_plan():
    """Get a safe execution validation plan."""
    scan = scanner.scan()
    plan = planner.plan(scan)
    risks = risk_analyzer.analyze(scan, plan)
    return SuccessResponse(message="Execution plan generated.", data={"plan": plan, "risks": risks})


@router.post("/preflight", response_model=SuccessResponse)
async def run_preflight():
    """Run full validation pipeline: scan -> plan -> preflight -> authorize."""
    orch = ValidationOrchestrator()
    result = orch.run_full()
    return SuccessResponse(
        message=f"Validation {'authorized' if result['orchestration']['status'] == 'AUTHORIZED' else 'failed'}",
        data=result,
    )


@router.post("/authorize", response_model=SuccessResponse)
async def authorize_execution():
    """Run full validation and return authorization decision."""
    orch = ValidationOrchestrator()
    result = orch.run_full()
    auth = result["authorization"]
    if not auth.get("authorized"):
        raise HTTPException(status_code=400, detail=f"Execution not authorized: {auth.get('restrictions', [])}")
    return SuccessResponse(message="Execution authorized.", data=auth)


@router.get("/authorization", response_model=SuccessResponse)
async def get_authorization():
    """Get the current authorization status."""
    auth_path = Path("data") / "execution_authorization.json"
    if not auth_path.exists():
        raise HTTPException(status_code=404, detail="No authorization found. Run /validation/preflight first.")
    import json
    with open(auth_path) as f:
        auth = json.load(f)
    return SuccessResponse(message="Authorization loaded.", data=auth)


@router.get("/state", response_model=SuccessResponse)
async def get_validation_state():
    """Get the current validation state."""
    orch = ValidationOrchestrator()
    return SuccessResponse(message=f"Validation state: {orch._sm.state}", data={"state": orch._sm.state})
