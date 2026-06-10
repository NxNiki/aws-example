"""FastAPI application factory for dashboard_api.

Mounts the data/health routers and, in production, serves the built React SPA
(``frontend/dist``) at ``/`` so a single container/ALB target serves both the
API and the UI. See docs/frontend_redesign.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dashboard_api.routers import data, health, views
from dashboard_api.settings import settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="dashboard_api", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(data.router)
    app.include_router(views.router)

    dist = Path(settings.frontend_dist)
    if dist.is_dir():
        # html=True serves index.html for unmatched paths (SPA client-side routing).
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="spa")
        logger.info("Serving SPA static assets from %s", dist)
    else:
        logger.info("No SPA build at %s — running API-only (use the Vite dev server for the UI)", dist)

    return app


app = create_app()
