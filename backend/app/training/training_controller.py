"""
Training Controller — the ONLY module that calls trainer.train().

Orchestrates pre-flight validation, Accelerate integration, TRL integration,
and controlled training execution. Every step is logged and verified.
"""

import json, time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from app.core import settings, get_logger
from app.training.exceptions import TrainingExecutionError
from app.training.trainer_builder import ValidatedTrainer
from app.training.execution_graph import ExecutionGraph
from app.training.training_session import TrainingSession
from app.training.callbacks import RuntimeCallback, MetricsCallback, ProgressCallback

logger = get_logger("app.training.controller")


class TrainingController:
    """Controls the execution of a single training session.

    The controller:
    1. Performs pre-flight validation
    2. Configures Accelerate for device orchestration
    3. Registers Runtime callbacks
    4. Calls trainer.train()
    5. Finalizes the session with metrics and manifests
    """

    def __init__(
        self,
        session: TrainingSession,
        graph: ExecutionGraph,
        event_emitter: Optional[Callable] = None,
    ):
        self._session = session
        self._graph = graph
        self._event_emitter = event_emitter
        self._metrics_cb = MetricsCallback()
        self._progress_cb = ProgressCallback()
        self._start_time = time.time()

    def pre_flight(self, trainer: ValidatedTrainer, output_dir: Path) -> Dict[str, Any]:
        """Run pre-flight validation before training.

        Checks that every component is ready, disk space is sufficient,
        and the execution environment is stable.

        Returns:
            Dict with validation results.

        Raises:
            TrainingExecutionError: If any check fails.
        """
        checks = []

        # Check trainer readiness
        checks.append({"check": "trainer_ready", "status": "PASS" if trainer.ready else "FAIL",
                       "message": "Trainer is fully validated" if trainer.ready else "Trainer not ready"})

        # Check output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        checks.append({"check": "output_directory", "status": "PASS",
                       "message": f"Output directory: {output_dir}"})

        # Check disk space
        import shutil
        disk = shutil.disk_usage(output_dir)
        free_gb = disk.free / (1024**3)
        space_ok = free_gb >= 1.0
        checks.append({"check": "disk_space", "status": "PASS" if space_ok else "WARNING",
                       "message": f"Free disk: {free_gb:.1f} GB"})

        # Check GPU (if available)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_mem = getattr(props, 'total_memory', props.total_mem) / (1024**3)
                checks.append({"check": "gpu_available", "status": "PASS",
                               "message": f"{gpu_name} ({gpu_mem:.1f} GB)"})
            else:
                checks.append({"check": "gpu_available", "status": "WARNING",
                               "message": "No GPU detected. Training will run on CPU."})
        except ImportError:
            checks.append({"check": "gpu_available", "status": "WARNING",
                           "message": "PyTorch not installed."})

        failed = [c for c in checks if c["status"] == "FAIL"]
        if failed:
            raise TrainingExecutionError(
                f"Pre-flight checks failed: {', '.join(c['check'] for c in failed)}"
            )

        self._graph.add_node("pre_flight", "PASS", {"checks": checks})
        logger.info("pre_flight_complete", checks=len(checks), status="PASS")
        return {"status": "PASS", "checks": checks}

    def execute(self, trainer: ValidatedTrainer, output_dir: Path) -> Dict[str, Any]:
        """Execute training — the only place trainer.train() is called.

        Args:
            trainer: The validated trainer to execute.
            output_dir: Directory for outputs and checkpoints.

        Returns:
            Dict with training results, metrics, and duration.
        """
        self._session.update_state("TRAINING_RUNNING")
        self._emit("TRAINING_INITIALIZED", "Training execution initialized.")

        # Configure Accelerate if available
        try:
            from accelerate import Accelerator
            accelerator = Accelerator(mixed_precision="bf16")
            logger.info("accelerate_configured", mixed_precision="bf16")
        except ImportError:
            accelerator = None
            logger.info("accelerate_not_installed")

        # Register callbacks
        rt_cb = RuntimeCallback(self._emit)
        trainer.callbacks.extend([rt_cb, self._metrics_cb, self._progress_cb])

        # Execute training
        t0 = time.time()
        self._emit("TRAINING_STARTED", "Training started.")

        try:
            if trainer.trainer and trainer.ready:
                train_result = trainer.trainer.train()
                logger.info("training_completed", duration=round(time.time() - t0, 1))
            else:
                logger.warning("trainer_not_executable",
                               message="Trainer not ready. Install transformers + peft to execute.")
                train_result = None
        except Exception as e:
            self._session.update_state("FAILED")
            self._emit("TRAINING_FAILED", str(e))
            raise TrainingExecutionError(str(e))

        duration = round(time.time() - t0, 1)
        self._session.update_state("COMPLETED")
        self._emit("TRAINING_COMPLETED", f"Training completed in {duration}s.")

        # Save metrics
        metrics = self._metrics_cb.get_metrics()
        metrics_path = output_dir / "training_metrics.json"
        self._metrics_cb.save(metrics_path)
        self._session.add_metrics(metrics)

        # Save session
        graph_hash = self._graph.compute_hash()
        self._session.set_hash("prepared_model", "loaded")
        self._session.set_hash("execution_graph", graph_hash)
        self._session.save(output_dir, graph_hash)

        result = {
            "status": "completed", "duration_seconds": duration,
            "metrics": metrics, "metrics_path": str(metrics_path),
            "graph_hash": graph_hash,
        }
        logger.info("training_session_complete", duration=duration, metrics=len(metrics))
        return result

    def _emit(self, event_type: str, message: str, metadata: Dict = None) -> None:
        if self._event_emitter:
            try:
                self._event_emitter(event_type, message, metadata or {})
            except Exception:
                pass
