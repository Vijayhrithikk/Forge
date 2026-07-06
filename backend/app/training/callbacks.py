"""
Training Callbacks — transformers-compatible callback infrastructure.

Callbacks emit Runtime Events during training. In transformers 5.x,
callbacks should subclass TrainerCallback for full compatibility.
"""

import time, json
from typing import Any, Optional, Callable
from app.core import get_logger

logger = get_logger("app.training.callbacks")


class RuntimeCallback:
    """Base callback that emits Runtime Events. Compatible with transformers 4.x and 5.x."""
    def __init__(self, event_emitter: Optional[Callable] = None):
        self._emit = event_emitter or (lambda *a, **kw: None)

    def on_train_begin(self, args, state=None, control=None, **kwargs):
        self._emit("TRAINING_STARTED", "Training started.")

    def on_epoch_begin(self, args, state=None, control=None, **kwargs):
        self._emit("EPOCH_STARTED", f"Epoch {args.epoch + 1} started.")

    def on_step_end(self, args, state=None, control=None, **kwargs):
        if args.global_step % 10 == 0:
            self._emit("STEP_COMPLETED", f"Step {args.global_step}.")

    def on_log(self, args, state=None, control=None, logs=None, **kwargs):
        if logs:
            self._emit("LOGGING", "Metrics logged.", logs)

    def on_epoch_end(self, args, state=None, control=None, **kwargs):
        self._emit("EPOCH_COMPLETED", f"Epoch {args.epoch + 1} completed.")

    def on_train_end(self, args, state=None, control=None, **kwargs):
        self._emit("TRAINING_COMPLETED", "Training completed.")


class MetricsCallback:
    """Collects training metrics. Compatible with transformers 4.x and 5.x."""
    def __init__(self):
        self.metrics: list = []
        self._start_time = time.time()

    def on_log(self, args, state=None, control=None, logs=None, **kwargs):
        """Transformers 5.x passes logs as keyword argument."""
        logs = logs or kwargs.get('logs', {})
        if logs:
            entry = {
                "epoch": round(getattr(args, 'epoch', 0) or 0, 2),
                "global_step": getattr(args, 'global_step', 0) or 0,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "elapsed": round(time.time() - self._start_time, 1),
            }
            self.metrics.append(entry)

    def get_metrics(self) -> list:
        return self.metrics

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)


class ProgressCallback:
    """Tracks training progress. Compatible with transformers 4.x and 5.x."""
    def __init__(self, total_steps: int = 0):
        self.total_steps = total_steps
        self.current_step = 0

    def on_step_end(self, args, state=None, control=None, **kwargs):
        self.current_step = getattr(args, 'global_step', 0) or 0

    @property
    def progress_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100)
