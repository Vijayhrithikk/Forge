"""
Training Callbacks — Runtime-integrated callback infrastructure.

Callbacks emit Runtime Events during training. They never modify
training logic. Supports logging, metrics, progress, and checkpoint hooks.
"""

import time
from typing import Dict, Any, Optional, Callable
from app.core import get_logger

logger = get_logger("app.training.callbacks")


class RuntimeCallback:
    """Base callback that emits Runtime Events."""
    def __init__(self, event_emitter: Optional[Callable] = None):
        self._emit = event_emitter or (lambda *a, **kw: None)

    def on_train_begin(self, args):
        self._emit("TRAINING_STARTED", "Training started.")

    def on_epoch_begin(self, args):
        self._emit("EPOCH_STARTED", f"Epoch {args.epoch + 1} started.")

    def on_step_end(self, args):
        if args.global_step % 10 == 0:
            self._emit("STEP_COMPLETED", f"Step {args.global_step}.")

    def on_log(self, args, logs):
        self._emit("LOGGING", "Metrics logged.", logs)

    def on_epoch_end(self, args):
        self._emit("EPOCH_COMPLETED", f"Epoch {args.epoch + 1} completed.")

    def on_train_end(self, args):
        self._emit("TRAINING_COMPLETED", "Training completed.")


class MetricsCallback:
    """Collects and emits training metrics."""
    def __init__(self):
        self.metrics: list = []
        self._start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {
                "epoch": round(args.epoch, 2) if args else 0,
                "global_step": args.global_step if args else 0,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "elapsed": round(time.time() - self._start_time, 1),
            }
            self.metrics.append(entry)

    def get_metrics(self) -> list:
        return self.metrics

    def save(self, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)


class ProgressCallback:
    """Tracks training progress for the Runtime."""
    def __init__(self, total_steps: int = 0):
        self.total_steps = total_steps
        self.current_step = 0

    def on_step_end(self, args):
        self.current_step = args.global_step

    @property
    def progress_pct(self) -> float:
        if self.total_steps == 0: return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100)
