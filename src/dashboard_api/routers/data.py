"""Data API: per-game configs and metric time-series for the React dashboard."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from dashboard_api.schemas.data import (
    ConfigDetail,
    ConfigList,
    CorrMatrix,
    DateBounds,
    DeepdiveMetrics,
    DeepdiveRequest,
    DeepdiveResponse,
    Granularity,
    GroupDistributionRequest,
    GroupDistributionResponse,
    GroupStat,
    GroupValues,
    HistogramSeries,
    ScatterSeries,
    Series,
    SeriesRequest,
    SeriesResponse,
)
from dashboard_api.services.common import SeriesError
from dashboard_api.services.configs import build_config_detail, list_config_summaries, load_raw_config
from dashboard_api.services.deepdive import load_deepdive, load_deepdive_metrics
from dashboard_api.services.group_distribution import load_group_distribution
from dashboard_api.services.series import load_date_bounds, load_group_values, load_series
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
        date_col, series, missing = load_series(
            cfg, req.granularity, req.metrics, req.date_from, req.date_to, req.group_values
        )
    except SeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # surface S3/credential/read failures as 503 rather than a 500 stack trace
        logger.exception("series read failed for config=%s", req.config)
        raise HTTPException(status_code=503, detail=f"Failed to read series data: {exc}") from exc

    return SeriesResponse(
        config=req.config,
        granularity=req.granularity,
        date_col=date_col,
        series=[Series(**s) for s in series],
        missing=missing,
    )


@router.get("/group-values", response_model=GroupValues)
def get_group_values(config: str, granularity: Granularity = "day") -> GroupValues:
    """API endpoint: GET /api/data/group-values — distinct cohort values per
    user-group column, for the Stats-by-Date cohort pickers.

    Returns ``{}`` when the config defines no user_group columns. Reads the same
    user-level parquet as ``/series``.
    """
    cfg = load_raw_config(settings.config_dir, config)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown config '{config}'")
    try:
        values = load_group_values(cfg, granularity)
    except SeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("group-values read failed for config=%s", config)
        raise HTTPException(status_code=503, detail=f"Failed to read group values: {exc}") from exc

    return GroupValues(config=config, granularity=granularity, values=values)


@router.get("/date-bounds", response_model=DateBounds)
def get_date_bounds(config: str, granularity: Granularity = "day") -> DateBounds:
    """API endpoint: GET /api/data/date-bounds — min/max date in the source data.

    The Stats-by-Date tab seeds its default window (last 30 days) from ``max``.
    """
    cfg = load_raw_config(settings.config_dir, config)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown config '{config}'")
    try:
        lo, hi = load_date_bounds(cfg, granularity)
    except SeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("date-bounds read failed for config=%s", config)
        raise HTTPException(status_code=503, detail=f"Failed to read date bounds: {exc}") from exc

    return DateBounds(config=config, granularity=granularity, min=lo, max=hi)


@router.post("/group-distribution", response_model=GroupDistributionResponse)
def post_group_distribution(req: GroupDistributionRequest) -> GroupDistributionResponse:
    """API endpoint: POST /api/data/group-distribution — Stats-by-Group tab.

    Per (cohort × date-range), the metric's distribution as a 5-number summary +
    mean/bootstrap-CI, so the frontend renders a box plot or a bar chart.
    """
    cfg = load_raw_config(settings.config_dir, req.config)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown config '{req.config}'")
    try:
        stats, missing = load_group_distribution(
            cfg,
            req.granularity,
            req.metric,
            [(r.start, r.end) for r in req.ranges],
            req.group_values,
            req.clip.enable,
            req.clip.min,
            req.clip.max,
        )
    except SeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("group-distribution read failed for config=%s", req.config)
        raise HTTPException(status_code=503, detail=f"Failed to read group distribution: {exc}") from exc

    return GroupDistributionResponse(
        config=req.config,
        granularity=req.granularity,
        metric=req.metric,
        stats=[GroupStat(**s) for s in stats],
        missing=missing,
    )


@router.post("/deepdive", response_model=DeepdiveResponse)
def post_deepdive(req: DeepdiveRequest) -> DeepdiveResponse:
    """API endpoint: POST /api/data/deepdive — Deep Dive tab.

    Pre-binned histograms or a Pearson correlation matrix over the selected
    metrics, per (cohort × date-range), for the derived or user-level panel.
    """
    cfg = load_raw_config(settings.config_dir, req.config)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown config '{req.config}'")
    try:
        histograms, heatmaps, scatters, missing = load_deepdive(
            cfg,
            req.granularity,
            req.panel,
            req.mode,
            req.metrics,
            [(r.start, r.end) for r in req.ranges],
            req.group_values,
            req.clip.enable,
            req.clip.min,
            req.clip.max,
            req.nbins,
            req.normalize,
            req.outliers_std,
        )
    except SeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("deepdive read failed for config=%s", req.config)
        raise HTTPException(status_code=503, detail=f"Failed to read deepdive data: {exc}") from exc

    return DeepdiveResponse(
        config=req.config,
        granularity=req.granularity,
        panel=req.panel,
        mode=req.mode,
        histograms=[HistogramSeries(**h) for h in histograms],
        heatmaps=[CorrMatrix(**m) for m in heatmaps],
        scatters=[ScatterSeries(**s) for s in scatters],
        missing=missing,
    )


@router.get("/deepdive-metrics", response_model=DeepdiveMetrics)
def get_deepdive_metrics(config: str, granularity: Granularity = "day") -> DeepdiveMetrics:
    """API endpoint: GET /api/data/deepdive-metrics — metric lists for the Deep
    Dive panel pickers (derived = group-level DataMetrics, user = raw user_* columns).
    """
    cfg = load_raw_config(settings.config_dir, config)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown config '{config}'")
    try:
        derived, user = load_deepdive_metrics(cfg, granularity)
    except SeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("deepdive-metrics read failed for config=%s", config)
        raise HTTPException(status_code=503, detail=f"Failed to read deepdive metrics: {exc}") from exc

    return DeepdiveMetrics(config=config, granularity=granularity, derived=derived, user=user)
