"""
Training API — endpoints for the Training Runtime.

Mission C: Create training sessions, execute training, query state.
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query

from app.core import get_logger
from app.schemas.responses import SuccessResponse
from app.engines.dataset.workspace import workspace_engine
from app.training.dataset_runtime import DatasetRuntime
from app.training.tokenization import TokenizationRuntime
from app.training.peft_runtime import PEFTRuntime
from app.training.trainer_builder import TrainerBuilder
from app.training.execution_graph import ExecutionGraph
from app.training.training_session import TrainingSession
from app.training.training_controller import TrainingController

router = APIRouter(prefix="/runtime", tags=["training"])
logger = get_logger("app.api.training")

_active_sessions: dict[str, TrainingSession] = {}


@router.post("/train", response_model=SuccessResponse)
async def start_training(project_id: str = Query(...)):
    """Execute the full training pipeline: dataset -> tokenize -> PEFT -> trainer -> execute.

    This is the primary training endpoint. It performs pre-flight validation,
    builds the trainer, and executes training under Runtime supervision.
    """
    try:
        project_path = workspace_engine.get_project_path(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    # Load training plan
    plan_path = project_path / "reports" / "training_plan.json"
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="No training plan. Generate one from Configuration Studio.")
    with open(plan_path) as f:
        plan = json.load(f)

    runtime_id = f"run_{project_id[:8]}"
    runtime_dir = project_path / "runtime"
    output_dir = runtime_dir / "exports"

    # 1. Dataset Runtime
    ds = DatasetRuntime(project_path)
    ds_result = ds.load_and_validate(plan)
    ds.save_manifest(runtime_dir)

    # 2. Tokenization
    tok = TokenizationRuntime(max_length=plan.get("hyperparameters", {}).get("max_sequence_length", 2048))
    tok_result = tok.tokenize(ds_result["records"], plan)
    tok.save_report(runtime_dir)

    # 3. PEFT Runtime (if libraries available)
    peft = PEFTRuntime()
    peft_result = peft.inject(None, plan.get("lora", {}), plan.get("lora", {}).get("target_modules"))
    peft.save_manifest(peft_result["report"], runtime_dir)
    peft.save_trainable_report(peft_result["report"], runtime_dir)

    # 4. Trainer Builder
    builder = TrainerBuilder()
    trainer = builder.build(
        peft_model=peft_result["peft_model"],
        tokenizer=None,  # Would come from PreparedModel
        tokenized_dataset=tok_result["tokenized"],
        training_plan=plan,
        output_dir=output_dir,
    )

    # 5. Execution Graph + Session
    graph = ExecutionGraph(runtime_id)
    graph.build_default()
    graph.add_node("dataset_runtime", "READY", {"samples": ds_result["sample_count"]})
    graph.add_node("tokenization_runtime", "READY", {"total_tokens": tok_result["total_tokens"]})
    graph.add_node("peft_runtime", "READY", {"trainable": peft_result["report"]["trainable_params"]})
    graph.add_node("trainer_builder", "READY" if trainer.ready else "PENDING")

    session = TrainingSession(runtime_id, project_id)
    session.set_hash("dataset", ds_result["hash"])
    session.set_hash("execution_graph", graph.compute_hash())

    graph.save(runtime_dir)
    session.save(runtime_dir)

    _active_sessions[project_id] = session

    # 6. Execute training
    controller = TrainingController(session, graph)
    pre_flight = controller.pre_flight(trainer, output_dir)
    train_result = controller.execute(trainer, output_dir)

    return SuccessResponse(
        message=f"Training {'completed' if train_result['status'] == 'completed' else 'attempted'}.",
        data={
            "session_id": session.session_id,
            "runtime_id": runtime_id,
            "pre_flight": pre_flight,
            "training": train_result,
            "dataset": {"samples": ds_result["sample_count"]},
            "tokenization": {"total_tokens": tok_result["total_tokens"]},
            "peft": peft_result["report"],
            "trainer_hash": trainer.trainer_hash,
        },
    )


@router.get("/training-session", response_model=SuccessResponse)
async def get_training_session(project_id: str = Query(...)):
    """Get the current training session status."""
    session = _active_sessions.get(project_id)
    if not session:
        # Try loading from disk
        try:
            session_path = workspace_engine.get_project_path(project_id) / "runtime" / "training_session.json"
            if session_path.exists():
                with open(session_path) as f:
                    return SuccessResponse(message="Session loaded from disk.", data=json.load(f))
        except Exception:
            pass
        raise HTTPException(status_code=404, detail="No active training session.")

    return SuccessResponse(message="Training session.", data=session.generate())


@router.get("/training-metrics", response_model=SuccessResponse)
async def get_training_metrics(project_id: str = Query(...)):
    """Get training metrics from the last training run."""
    try:
        metrics_path = workspace_engine.get_project_path(project_id) / "runtime" / "exports" / "training_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            return SuccessResponse(message=f"{len(metrics)} metric entries.", data={"metrics": metrics})
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="No training metrics found.")


@router.get("/trainable-parameters", response_model=SuccessResponse)
async def get_trainable_parameters(project_id: str = Query(...)):
    """Get the trainable parameters report."""
    path = workspace_engine.get_project_path(project_id) / "runtime" / "trainable_parameters.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No trainable parameters report.")
    with open(path) as f:
        return SuccessResponse(message="Trainable parameters report.", data=json.load(f))


@router.get("/execution-graph", response_model=SuccessResponse)
async def get_execution_graph(project_id: str = Query(...)):
    """Get the training execution graph."""
    path = workspace_engine.get_project_path(project_id) / "runtime" / "execution_graph.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No execution graph.")
    with open(path) as f:
        return SuccessResponse(message="Execution graph.", data=json.load(f))


@router.get("/training-state", response_model=SuccessResponse)
async def get_training_state(project_id: str = Query(...)):
    """Get the current training state summary."""
    runtime_dir = workspace_engine.get_project_path(project_id) / "runtime"
    state = {"project_id": project_id, "artifacts": {}}
    for name in ["training_session.json", "execution_graph.json", "trainable_parameters.json",
                 "dataset_runtime_manifest.json", "peft_manifest.json", "training_metrics.json"]:
        p = runtime_dir / "exports" / name if name == "training_metrics.json" else runtime_dir / name
        state["artifacts"][name] = "present" if p.exists() else "missing"
    return SuccessResponse(message="Training state summary.", data=state)
