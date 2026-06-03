"""Runtime settings for dashboard_api, resolved from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_config_dir() -> str:
    # src/dashboard_api/settings.py -> src/dashboards
    return str(Path(__file__).resolve().parent.parent / "dashboards")


def _default_frontend_dist() -> str:
    # repo_root/frontend/dist (present only after `vite build`)
    return str(Path(__file__).resolve().parents[2] / "frontend" / "dist")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    config_dir: str = field(default_factory=lambda: os.environ.get("DASHBOARD_CONFIG_DIR", _default_config_dir()))
    frontend_dist: str = field(
        default_factory=lambda: os.environ.get("DASHBOARD_FRONTEND_DIST", _default_frontend_dist())
    )
    agent_api_url: str = field(default_factory=lambda: os.environ.get("AGENT_API_URL", "http://localhost:8051"))
    cors_origins: list[str] = field(
        default_factory=lambda: _split_csv(os.environ.get("DASHBOARD_API_CORS", "http://localhost:5173"))
    )


settings = Settings()
