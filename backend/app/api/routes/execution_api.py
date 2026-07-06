"""Execution API — target detection, session management, real execution, replay, history."""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from app.core import get_logger
from app.schemas.responses import SuccessResponse
from app.execution.targets import target_info
from app.execution.controller import ExecutionController
from app.execution.matrix import ValidationMatrix
from app.execution.consistency import validate_consistency
from app.execution.replay import reconstruct_from_session
from app.execution.history import ValidationHistory

router = APIRouter(prefix="/execution", tags=["execution"])
logger = get_logger("app.api.execution")

_controller: ExecutionController = None


def _get_controller() -> ExecutionController:
    global _controller
    if not _controller: _controller = ExecutionController()
    return _controller


@router.get("/target", response_model=SuccessResponse)
async def get_target():
    return SuccessResponse(message="Execution target detected.", data=target_info())


@router.get("/session", response_model=SuccessResponse)
async def get_session():
    ctrl = _get_controller()
    session = ctrl.get_session()
    if not session:
        raise HTTPException(status_code=404, detail="No execution session. Run /execution/prepare first.")
    return SuccessResponse(message="Execution session.", data=session)


@router.get("/certificate", response_model=SuccessResponse)
async def get_certificate():
    path = Path("data/execution/execution_certificate.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No certificate. Run /execution/run first.")
    with open(path) as f: cert = json.load(f)
    return SuccessResponse(message=f"Certificate: {cert['overall_result']}", data=cert)


@router.post("/prepare", response_model=SuccessResponse)
async def prepare_execution():
    ctrl = _get_controller()
    result = ctrl.prepare()
    return SuccessResponse(message="Execution prepared.", data=result)


@router.post("/run", response_model=SuccessResponse)
async def run_execution():
    ctrl = _get_controller()
    result = ctrl.execute()
    return SuccessResponse(message=f"Execution: {result['status']}", data=result)


@router.get("/trace", response_model=SuccessResponse)
async def get_trace():
    path = Path("data/execution/execution_trace.json")
    if not path.exists(): raise HTTPException(status_code=404, detail="No trace. Run /execution/run first.")
    with open(path) as f: trace = json.load(f)
    return SuccessResponse(message="Execution trace.", data=trace)


@router.get("/history", response_model=SuccessResponse)
async def get_history():
    history = ValidationHistory()
    executions = history.list_executions()
    return SuccessResponse(message=f"{len(executions)} execution(s).", data={"executions": executions})


@router.get("/consistency", response_model=SuccessResponse)
async def get_consistency():
    report = validate_consistency(Path("data/execution"))
    return SuccessResponse(message=f"Consistency: {report['overall']}", data=report)


@router.post("/replay", response_model=SuccessResponse)
async def replay_execution():
    path = Path("data/execution/execution_session.json")
    if not path.exists(): raise HTTPException(status_code=404, detail="No session to replay.")
    result = reconstruct_from_session(path)
    return SuccessResponse(message="Execution replayed.", data=result)
