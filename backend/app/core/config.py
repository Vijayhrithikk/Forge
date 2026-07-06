"""
Application configuration loaded from environment variables.

All configurable values have sensible defaults for development.
Production overrides are expected via environment variables.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Central application configuration.

    Values are loaded from environment variables with fallback defaults.
    Prefix APP_ is used for application-level settings.
    """

    # Application
    app_name: str = "Forge"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True

    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_workers: int = 1

    # CORS
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Paths
    forge_data_dir: str = "./data"

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == "development"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton instance
settings = Settings()
