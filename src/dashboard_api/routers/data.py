"""Data API: per-game configs and metric time-series for the React dashboard."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

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
    LifecycleGroup,
    ScatterSeries,
    Series,
    SeriesRequest,
    SeriesResponse,
    SummaryCell,
    SummaryColumn,
    SummaryRow,
    SummaryTableRequest,
    SummaryTableResponse,
)
from dashboard_api.services.common import SeriesError, cached_response
from dashboard_api.services.configs import build_config_detail, list_config_summaries, load_raw_config
from dashboard_api.services.deepdive import load_deepdive, load_deepdive_metrics
from dashboard_api.services.group_distribution import load_group_distribution
from dashboard_api.services.series import load_date_bounds, load_group_values, load_group_values_in_range, load_series
from dashboard_api.services.summary_table import load_summary_table
from dashboard_api.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])


def _lifecycle(groups: Optional[list[LifecycleGroup]]) -> Optional[list[dict]]:
    return [g.model_dump() for g in groups] if groups else None


def _ranges_dims(groups) -> Optional[list[dict]]:
    return [g.model_dump() for g in groups] if groups else None


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
        # Identical concurrent/repeated requests (several users on the same
        # view) compute once and share the result.
        # Empty series are not cached: they can be a read racing the ETL's
        # partition rewrite, and pinning them would blank panels for the TTL.
        date_col, series, missing = cached_response(
            "series:" + req.model_dump_json(),
            lambda: load_series(
                cfg,
                req.granularity,
                req.metrics,
                req.date_from,
                req.date_to,
                req.group_values,
                lifecycle=_lifecycle(req.lifecycle_groups),
                range_groups=_ranges_dims(req.range_groups),
                ranges=[(r.start, r.end) for r in req.ranges] if req.ranges is not None else None,
            ),
            should_cache=lambda v: bool(v[1]),
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


# Full-vocabulary scans read every partition file, so cache them; the values
# only change when the daily ETL lands. Expired entries are served stale while
# a background thread refreshes, so no request ever waits on the full scan
# except the very first per (config, granularity) after a cold start.
# Availability (range-scoped) is cheap (period-pruned read) per request.
#
# All state below is guarded by _group_values_lock (request threadpool +
# refresh/warmup threads share it). Cold misses are single-flighted per key
# via _GROUP_VALUES_LOADING: the first caller scans, concurrent callers wait
# on its event — never run N identical full scans. The lock is NOT held
# during scans, so cached configs stay servable while another config loads.
_GROUP_VALUES_CACHE: dict[tuple[str, str], tuple[float, dict[str, list[str]]]] = {}
_GROUP_VALUES_REFRESHING: set[tuple[str, str]] = set()
_GROUP_VALUES_LOADING: dict[tuple[str, str], threading.Event] = {}
_GROUP_VALUES_TTL_SECONDS = 600.0
_group_values_lock = threading.Lock()


def warm_group_values_cache() -> None:
    """Fill the vocabulary cache for every config's day granularity (called
    from a startup thread, off the request path)."""
    for summary in list_config_summaries(settings.config_dir):
        cfg = load_raw_config(settings.config_dir, summary.id)
        if cfg is None:
            continue
        try:
            _cached_group_values(summary.id, cfg, "day")
            logger.info("group-values cache warmed for %s", summary.id)
        except Exception:
            logger.exception("group-values warmup failed for %s", summary.id)


def _cached_group_values(config: str, cfg: dict, granularity: str) -> dict[str, list[str]]:
    key = (config, granularity)
    while True:
        with _group_values_lock:
            cached = _GROUP_VALUES_CACHE.get(key)
            if cached is not None:
                break
            loading = _GROUP_VALUES_LOADING.get(key)
            if loading is None:
                loading = threading.Event()
                _GROUP_VALUES_LOADING[key] = loading
                is_loader = True
            else:
                is_loader = False
        if is_loader:
            try:
                values = load_group_values(cfg, granularity)
                with _group_values_lock:
                    _GROUP_VALUES_CACHE[key] = (time.monotonic(), values)
                return values
            finally:
                with _group_values_lock:
                    _GROUP_VALUES_LOADING.pop(key, None)
                loading.set()
        else:
            # Re-check the cache after the loader finishes; if it failed, the
            # next iteration elects this thread as the loader (retry).
            loading.wait()

    spawn_refresh = False
    with _group_values_lock:
        if time.monotonic() - cached[0] >= _GROUP_VALUES_TTL_SECONDS and key not in _GROUP_VALUES_REFRESHING:
            _GROUP_VALUES_REFRESHING.add(key)
            spawn_refresh = True
        result = cached[1]

    if spawn_refresh:

        def _refresh() -> None:
            try:
                values = load_group_values(cfg, granularity)
                with _group_values_lock:
                    _GROUP_VALUES_CACHE[key] = (time.monotonic(), values)
            except Exception:
                logger.exception("group-values background refresh failed for %s", key)
            finally:
                with _group_values_lock:
                    _GROUP_VALUES_REFRESHING.discard(key)

        threading.Thread(target=_refresh, name=f"group-values-{config}-{granularity}", daemon=True).start()
    return result


@router.get("/group-values", response_model=GroupValues)
def get_group_values(
    config: str,
    granularity: Granularity = "day",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> GroupValues:
    """API endpoint: GET /api/data/group-values — distinct cohort values per
    user-group column, for the Stats-by-Date cohort pickers.

    ``values`` is the full (cached) vocabulary so labels never disappear when
    the date range moves; with ``start``/``end`` the response also carries
    ``available`` — the subset present in that range — which the pickers use
    to gray out values with no data in view.

    Returns ``{}`` when the config defines no user_group columns. Reads the same
    user-level parquet as ``/series``.
    """
    cfg = load_raw_config(settings.config_dir, config)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown config '{config}'")
    try:
        values = _cached_group_values(config, cfg, granularity)
        available = None
        if start and end:
            available = load_group_values_in_range(cfg, granularity, start, end)
    except SeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date range: {exc}") from exc
    except Exception as exc:
        logger.exception("group-values read failed for config=%s", config)
        raise HTTPException(status_code=503, detail=f"Failed to read group values: {exc}") from exc

    return GroupValues(config=config, granularity=granularity, values=values, available=available)


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
        stats, missing = cached_response(
            "group-distribution:" + req.model_dump_json(),
            lambda: load_group_distribution(
                cfg,
                req.granularity,
                req.metric,
                [(r.start, r.end) for r in req.ranges],
                req.group_values,
                _lifecycle(req.lifecycle_groups),
                _ranges_dims(req.range_groups),
                req.clip.enable,
                req.clip.min,
                req.clip.max,
                req.filter.enable,
                req.filter.min,
                req.filter.max,
            ),
            should_cache=lambda v: bool(v[0]),
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


@router.post("/summary-table", response_model=SummaryTableResponse)
def post_summary_table(req: SummaryTableRequest) -> SummaryTableResponse:
    """API endpoint: POST /api/data/summary-table — Summary-table tab.

    One grid for all metrics at once: rows are metrics (grouped by the config's
    metric groups), columns are the selected cohort × date-range combinations.
    Each cell is a metric's mean/median/quartiles for that column; the frontend
    flags one column as the reference and renders the rest as ``value (±%)``.
    With ``pvalues=true`` each row also carries a Welch t-test (2 columns) or
    one-way ANOVA (3+) p-value.
    """
    cfg = load_raw_config(settings.config_dir, req.config)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown config '{req.config}'")
    try:
        columns, rows = load_summary_table(
            cfg,
            req.granularity,
            [(r.start, r.end) for r in req.ranges],
            req.group_values,
            _lifecycle(req.lifecycle_groups),
            _ranges_dims(req.range_groups),
            {
                m: {"log": o.log, "clip_enable": o.clip.enable, "clip_min": o.clip.min, "clip_max": o.clip.max}
                for m, o in req.metric_options.items()
            },
            req.pvalues,
        )
    except SeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("summary-table read failed for config=%s", req.config)
        raise HTTPException(status_code=503, detail=f"Failed to read summary table: {exc}") from exc

    return SummaryTableResponse(
        config=req.config,
        granularity=req.granularity,
        columns=[SummaryColumn(**c) for c in columns],
        rows=[
            SummaryRow(
                metric=r["metric"],
                group_id=r["group_id"],
                group_label=r["group_label"],
                cells=[SummaryCell(**c) if c is not None else None for c in r["cells"]],
                pvalue=r.get("pvalue"),
                test=r.get("test"),
                missing=r["missing"],
            )
            for r in rows
        ],
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
            _lifecycle(req.lifecycle_groups),
            _ranges_dims(req.range_groups),
            req.clip.enable,
            req.clip.min,
            req.clip.max,
            req.filter.enable,
            req.filter.min,
            req.filter.max,
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
