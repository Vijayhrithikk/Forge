"""
Structured logging configuration for Forge.

Uses structlog to emit JSON-structured logs that are
machine-parseable and human-readable during development.
"""

import structlog
import logging
import sys
from typing import Any

from app.core.config import settings


def configure_logging() -> None:
    """Initialize structured logging for the application.

    In development mode, logs are pretty-printed to the console.
    In production, logs are emitted as JSON for log aggregation.
    """

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if settings.is_development and settings.log_format != "json":
        # Pretty-printed console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.rich_traceback,
            )
        ]
    else:
        # JSON output for production / structured log pipelines
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound logger with the component name set.

    Args:
        name: Component name (e.g. "app.api.health")

    Returns:
        A structlog BoundLogger with component context.
    """
    return structlog.get_logger().bind(component=name)
