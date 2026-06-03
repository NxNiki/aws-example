"""Data API: per-game configs and metric time-series for the React dashboard."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from dashboard_api.schemas.data import ConfigDetail, ConfigList, Series, SeriesRequest, SeriesResponse
from dashboard_api.services.configs import build_config_detail, list_config_summaries, load_raw_config
from dashboard_api.services.series import SeriesError, load_series
from dashboard_api.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/configs", response_model=ConfigList)
def list_configs() -> ConfigList:
    """API endpoint: GET /api/data/configs — powers the game/config picker in the SPA top bar.

    Returns one entry per ``dashboard_config-*.yaml`` in ``DASHBOARD_CONFIG_DIR``;
    the id is the filename minus the ``dashboard_config-`` prefix (e.g. ``ss01``,
    ``fishhunter``).
    """
    return ConfigList(configs=list_config_summaries(settings.config_dir))


@router.get("/config/{config_id}", response_model=ConfigDetail)
def get_config(config_id: str) -> ConfigDetail:
    """API endpoint: GET /api/data/config/{config_id} — resolved config for one game.

    Drives which tabs, metric groups, and date granularities the dashboard
    shows. This is also the snapshot the agent reads to decide which config to
    load and which tab to open for a natural-language data request.
    """
    cfg = load_raw_config(settings.config_dir, config_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown config '{config_id}'")
    return build_config_detail(config_id, cfg)


@router.post("/series", response_model=SeriesResponse)
def post_series(req: SeriesRequest) -> SeriesResponse:
    """API endpoint: POST /api/data/series — metric time-series for a chart.

    Returns date-aligned series the frontend encodes into an ECharts chart.
    Metrics not present in the source parquet come back in ``missing`` rather
    than failing the request, so the UI can flag them.
    """
    cfg = load_raw_config(settings.config_dir, req.config)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown config '{req.config}'")
    try:
        date_col, x, series, missing = load_series(cfg, req.granularity, req.metrics, req.date_from, req.date_to)
    except SeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # surface S3/credential/read failures as 503 rather than a 500 stack trace
        logger.exception("series read failed for config=%s", req.config)
        raise HTTPException(status_code=503, detail=f"Failed to read series data: {exc}") from exc

    return SeriesResponse(
        config=req.config,
        granularity=req.granularity,
        date_col=date_col,
        x=x,
        series=[Series(**s) for s in series],
        missing=missing,
    )
