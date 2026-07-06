"""
Runtime State Machine — strongly typed, explicit transitions.

Never use string literals for states. Every transition is validated.
Illegal transitions raise errors with explanations.
"""

from enum import Enum
from typing import Set, List
from datetime import datetime, timezone

from app.core import get_logger

logger = get_logger("app.runtime.state_machine")


class RuntimeState(str, Enum):
    """Every possible Runtime state.

    States are ordered by lifecycle progression. The Runtime
    must pass through states in the defined order — no skipping.
    """

    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    PREPARING = "PREPARING"
    READY = "READY"
    TRAINING = "TRAINING"         # Future: Mission B
    CHECKPOINTING = "CHECKPOINTING"  # Future: Mission B
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RECOVERABLE = "RECOVERABLE"
    CANCELLED = "CANCELLED"


# Legal transitions from each state
LEGAL_TRANSITIONS: dict[RuntimeState, Set[RuntimeState]] = {
    RuntimeState.CREATED:       {RuntimeState.VALIDATING, RuntimeState.FAILED, RuntimeState.CANCELLED},
    RuntimeState.VALIDATING:    {RuntimeState.PREPARING, RuntimeState.FAILED, RuntimeState.RECOVERABLE, RuntimeState.CANCELLED},
    RuntimeState.PREPARING:     {RuntimeState.READY, RuntimeState.FAILED, RuntimeState.RECOVERABLE, RuntimeState.CANCELLED},
    RuntimeState.READY:         {RuntimeState.TRAINING, RuntimeState.FAILED, RuntimeState.CANCELLED},
    RuntimeState.TRAINING:      {RuntimeState.CHECKPOINTING, RuntimeState.COMPLETED, RuntimeState.FAILED, RuntimeState.RECOVERABLE, RuntimeState.CANCELLED},
    RuntimeState.CHECKPOINTING: {RuntimeState.TRAINING, RuntimeState.COMPLETED, RuntimeState.FAILED},
    RuntimeState.COMPLETED:     set(),  # Terminal
    RuntimeState.FAILED:        {RuntimeState.RECOVERABLE, RuntimeState.CANCELLED},
    RuntimeState.RECOVERABLE:   {RuntimeState.VALIDATING, RuntimeState.CANCELLED},
    RuntimeState.CANCELLED:     set(),  # Terminal
}


class InvalidTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""
    def __init__(self, current: RuntimeState, target: RuntimeState):
        self.current = current
        self.target = target
        super().__init__(
            f"Illegal transition: {current.value} -> {target.value}. "
            f"Allowed from {current.value}: {[s.value for s in LEGAL_TRANSITIONS.get(current, set())]}"
        )


class StateMachine:
    """Manages Runtime state transitions with validation.

    Every transition is logged. Illegal transitions raise
    InvalidTransitionError immediately.
    """

    def __init__(self, initial_state: RuntimeState = RuntimeState.CREATED):
        self._state = initial_state
        self._history: List[tuple[RuntimeState, RuntimeState, str]] = []  # (from, to, timestamp)

    @property
    def current_state(self) -> RuntimeState:
        return self._state

    @property
    def history(self) -> List[dict]:
        return [
            {"from": f.value, "to": t.value, "timestamp": ts}
            for f, t, ts in self._history
        ]

    def can_transition(self, target: RuntimeState) -> bool:
        """Check if a transition is legal without executing it."""
        return target in LEGAL_TRANSITIONS.get(self._state, set())

    def transition(self, target: RuntimeState) -> None:
        """Execute a state transition.

        Args:
            target: The desired state.

        Raises:
            InvalidTransitionError: If the transition is illegal.
        """
        if not self.can_transition(target):
            raise InvalidTransitionError(self._state, target)

        old = self._state
        self._state = target
        ts = datetime.now(timezone.utc).isoformat()
        self._history.append((old, target, ts))

        logger.info(
            "state_transition",
            from_state=old.value,
            to_state=target.value,
            timestamp=ts,
        )

    def is_terminal(self) -> bool:
        """Check if the current state is terminal."""
        return len(LEGAL_TRANSITIONS.get(self._state, set())) == 0

    def is_active(self) -> bool:
        """Check if the runtime is still active (not terminal)."""
        return not self.is_terminal()
