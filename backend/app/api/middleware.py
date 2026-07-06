"""
Global error handling middleware for FastAPI.

Converts unhandled exceptions into structured ErrorResponse objects
so the frontend always receives consistent error shapes.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.schemas.responses import ErrorDetail, ErrorResponse
from app.core import get_logger

logger = get_logger("app.api.middleware")


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware that catches unhandled exceptions and returns
    structured error responses following the Forge error contract."""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            logger.error(
                "unhandled_exception",
                path=request.url.path,
                method=request.method,
                error_type=type(exc).__name__,
            )

            error_detail = ErrorDetail(
                error_code="INTERNAL_ERROR",
                title="Internal Server Error",
                description="An unexpected error occurred while processing your request.",
                recommendation="Please try again. If the problem persists, check the server logs.",
                recoverable=True,
                details={
                    "path": request.url.path,
                    "method": request.method,
                },
            )

            error_response = ErrorResponse(
                error=error_detail,
                message=str(exc) if __debug__ else "An unexpected error occurred",
            )

            return JSONResponse(
                status_code=500,
                content=error_response.model_dump(),
            )
