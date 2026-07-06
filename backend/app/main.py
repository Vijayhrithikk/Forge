"""
Forge Backend Application Entry Point.

Initializes the FastAPI application with:
- Structured logging
- CORS middleware
- Error handling middleware
- API routes
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import settings, configure_logging, get_logger
from app.api.middleware import ErrorHandlingMiddleware
from app.api.routes.health import router as health_router
from app.api.routes.projects import router as projects_router
from app.api.routes.upload import router as upload_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.tokenizer import router as tokenizer_router
from app.api.routes.models import router as models_router
from app.api.routes.training import router as training_router
from app.api.routes.runtime_api import router as runtime_router
from app.api.routes.model_runtime import router as model_runtime_router
from app.api.routes.training_api import router as training_api_router
from app.api.routes.validation_api import router as validation_api_router
from app.api.routes.execution_api import router as execution_api_router
from app.api.routes.performance_api import router as performance_api_router
from app.api.routes.provider_api import router as provider_api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Handles startup and shutdown events with structured logging.
    """
    log = get_logger("app.lifecycle")

    # Startup
    configure_logging()
    log.info(
        "application_starting",
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        host=settings.backend_host,
        port=settings.backend_port,
    )

    yield

    # Shutdown
    log.info(
        "application_shutdown",
        name=settings.app_name,
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-inspired LoRA fine-tuning platform",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handling
    app.add_middleware(ErrorHandlingMiddleware)

    # Routes
    app.include_router(health_router, prefix="")
    app.include_router(projects_router, prefix="")
    app.include_router(upload_router, prefix="")
    app.include_router(datasets_router, prefix="")
    app.include_router(tokenizer_router, prefix="")
    app.include_router(models_router, prefix="")
    app.include_router(training_router, prefix="")
    app.include_router(runtime_router, prefix="")
    app.include_router(model_runtime_router, prefix="")
    app.include_router(training_api_router, prefix="")
    app.include_router(validation_api_router, prefix="")
    app.include_router(execution_api_router, prefix="")
    app.include_router(performance_api_router, prefix="")
    app.include_router(provider_api_router, prefix="")

    return app


app = create_app()
