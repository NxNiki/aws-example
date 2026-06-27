"""Dashboard view-snapshot API: save / list / load named UI setups.

The React-era replacement for the legacy Dash "save/load config" feature. The
snapshot body is opaque JSON owned by the frontend store; see services/views.py.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from dashboard_api.schemas.data import ViewList, ViewSaveResult
from dashboard_api.services.views import ViewError, list_views, load_view, save_view

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/views", tags=["views"])


@router.get("", response_model=ViewList)
def get_views() -> ViewList:
    """API endpoint: GET /api/views — list saved dashboard view snapshots."""
    try:
        return ViewList(views=list_views())
    except Exception as exc:
        logger.exception("list views failed")
        raise HTTPException(status_code=503, detail=f"Failed to list views: {exc}") from exc


@router.get("/{name}")
def get_view(name: str) -> dict[str, Any]:
    """API endpoint: GET /api/views/{name} — load one saved view snapshot (opaque JSON)."""
    try:
        return load_view(name)
    except ViewError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("load view failed for name=%s", name)
        raise HTTPException(status_code=404, detail=f"View '{name}' not found: {exc}") from exc


@router.put("/{name}", response_model=ViewSaveResult)
def put_view(name: str, snapshot: dict[str, Any] = Body(...)) -> ViewSaveResult:
    """API endpoint: PUT /api/views/{name} — save a dashboard view snapshot.

    Returns the sanitized name, the s3:// path written (surfaced to the user),
    and the refreshed view list.
    """
    try:
        path = save_view(name, snapshot)
        saved_name = path.rsplit("/", 1)[-1][: -len(".json")]
        return ViewSaveResult(name=saved_name, path=path, views=list_views())
    except ViewError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("save view failed for name=%s", name)
        raise HTTPException(status_code=503, detail=f"Failed to save view: {exc}") from exc
