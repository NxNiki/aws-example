"""FastAPI application factory for dashboard_api.

Mounts the data/health routers and, in production, serves the built React SPA
(``frontend/dist``) at ``/`` so a single container/ALB target serves both the
API and the UI. See docs/frontend_redesign.md.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dashboard_api.routers import data, health, report, views
from dashboard_api.settings import settings

logger = logging.getLogger(__name__)


def _install_request_metric(app: FastAPI) -> None:
    """Emit CloudWatch ``Dashboard/UserRequestCount`` per user request.

    Deployment feature: the scale-to-zero idle signal the legacy Dash app
    emitted from a Flask after-request hook — the ECS scale-in alarm watches
    this metric (Service dimension = ``DASHBOARD_SERVICE_NAME``). Active only
    when that env var is set (the ECS task definition sets it; local dev
    doesn't). Health checks are excluded so an idle service can reach zero;
    emission is fire-and-forget on a single worker thread and never fails a
    request.
    """
    service_name = os.environ.get("DASHBOARD_SERVICE_NAME")
    if not service_name:
        return

    import boto3

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
    client = boto3.client("cloudwatch", region_name=region)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cw-metric")

    def _emit() -> None:
        try:
            client.put_metric_data(
                Namespace="Dashboard",
                MetricData=[
                    {
                        "MetricName": "UserRequestCount",
                        "Dimensions": [{"Name": "Service", "Value": service_name}],
                        "Value": 1,
                        "Unit": "Count",
                    }
                ],
            )
        except Exception:  # noqa: BLE001 — metrics must never break a request
            logger.debug("UserRequestCount emission failed", exc_info=True)

    @app.middleware("http")
    async def emit_user_request_metric(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        if request.url.path != "/api/health":
            pool.submit(_emit)
        return response

    logger.info("UserRequestCount metric enabled (Service=%s)", service_name)


def create_app() -> FastAPI:
    app = FastAPI(title="dashboard_api", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _install_request_metric(app)

    app.include_router(health.router)
    app.include_router(data.router)
    app.include_router(views.router)
    app.include_router(report.router)

    @app.on_event("startup")
    def _warm_group_values() -> None:
        # Pre-fill the group-values vocabulary cache so no user's first picker
        # load waits on the full parquet scan; runs off the request path.
        threading.Thread(target=data.warm_group_values_cache, name="group-values-warmup", daemon=True).start()

    dist = Path(settings.frontend_dist)
    if dist.is_dir():
        # html=True serves index.html for unmatched paths (SPA client-side routing).
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="spa")
        logger.info("Serving SPA static assets from %s", dist)
    else:
        logger.info("No SPA build at %s — running API-only (use the Vite dev server for the UI)", dist)

    return app


app = create_app()
