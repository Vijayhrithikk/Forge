"""
Standardized API response models.

Every API response follows one of two shapes:
- SuccessResponse: for successful operations
- ErrorResponse: for failures

This contract ensures the frontend always knows what to expect.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


class SuccessResponse(BaseModel):
    """Standard success envelope for all API responses."""

    status: str = Field(default="success", description="Always 'success'")
    message: str = Field(
        default="Operation completed successfully",
        description="Human-readable success message",
    )
    data: Any = Field(default=None, description="Response payload")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp",
    )


class ErrorDetail(BaseModel):
    """Structured error information."""

    error_code: str = Field(description="Machine-readable error code")
    title: str = Field(description="Short human-readable error title")
    description: str = Field(description="Detailed error explanation")
    recommendation: str = Field(description="Recommended recovery action")
    recoverable: bool = Field(
        default=False, description="Whether the operation can be retried"
    )
    details: Optional[Any] = Field(
        default=None, description="Optional technical details"
    )


class ErrorResponse(BaseModel):
    """Standard error envelope for all API error responses."""

    status: str = Field(default="error", description="Always 'error'")
    error: ErrorDetail = Field(description="Structured error information")
    message: str = Field(description="Human-readable error summary")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp",
    )


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str = Field(description="Health status: healthy or unhealthy")
    application: str = Field(description="Application name")
    version: str = Field(description="Application version")
    build: str = Field(description="Build identifier or timestamp")
    git_commit: str = Field(description="Short git commit hash")
    python_version: str = Field(description="Python runtime version")
    supported_models: list[str] = Field(
        description="List of supported base model identifiers"
    )
    environment: str = Field(description="Current environment")
    uptime: float = Field(description="Server uptime in seconds")
    timestamp: str = Field(description="ISO-8601 timestamp of the health check")
