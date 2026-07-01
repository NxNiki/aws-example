"""Discover and parse the dashboard_config-*.yaml files into API models.

Wraps `bituslabs_ds.dashboard_utils.load_config` so the React frontend gets
the same per-game config (metrics, groups, granularities, tabs) the legacy
Dash app reads, without depending on the Dash app.

``config_dir`` (``settings.config_dir`` / ``DASHBOARD_CONFIG_DIR``) may be either a
local directory or an ``s3://bucket/prefix`` URI. When it is an S3 URI the configs
are read straight from the bucket, so adding a new ``dashboard_config-*.yaml`` only
needs an S3 upload — no dashboard image rebuild / redeploy. Reads are cached for a
short TTL (``DASHBOARD_CONFIG_CACHE_TTL`` seconds, default 60) so the per-request API
calls don't hit S3 every time.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from bituslabs_ds.dashboard_utils import load_config
from bituslabs_ds.s3_utils import is_s3_path, list_s3_files, parse_s3_path, read_yaml_from_s3, uri_basename
from dashboard_api.schemas.data import ConfigDetail, ConfigSummary, MetricGroup

logger = logging.getLogger(__name__)

CONFIG_PREFIX = "dashboard_config-"
_GRANULARITIES = ("day", "week", "month")

# Short TTL cache so listing/parsing configs (especially from S3) isn't repeated on
# every API request. Set DASHBOARD_CONFIG_CACHE_TTL=0 to disable (e.g. in tests).
_CACHE_TTL_SECONDS = float(os.environ.get("DASHBOARD_CONFIG_CACHE_TTL", "60"))
_cache: dict[str, tuple[float, Any]] = {}


def clear_config_cache() -> None:
    """Drop the config list/parse cache (used by tests, or to force a refresh)."""
    _cache.clear()


def _cached(key: str, producer: Callable[[], Any]) -> Any:
    if _CACHE_TTL_SECONDS <= 0:
        return producer()
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None and now - hit[0] < _CACHE_TTL_SECONDS:
        return hit[1]
    value = producer()
    _cache[key] = (now, value)
    return value


def _config_id(uri: str) -> str:
    name = uri_basename(uri)
    stem = name[: -len(".yaml")] if name.endswith(".yaml") else name
    return stem[len(CONFIG_PREFIX) :] if stem.startswith(CONFIG_PREFIX) else stem


def list_config_files(config_dir: str) -> list[str]:
    """All ``dashboard_config-*.yaml`` sources under ``config_dir``.

    Returns local paths or ``s3://`` URIs (strings) depending on ``config_dir``.
    """

    def _produce() -> list[str]:
        if is_s3_path(config_dir):
            bucket, prefix = parse_s3_path(config_dir)
            prefix = (prefix.rstrip("/") + "/") if prefix else ""
            # ``[^/]*`` keeps it to files directly under the prefix (no nested keys).
            pattern = rf"{re.escape(CONFIG_PREFIX)}[^/]*\.yaml$"
            return sorted(list_s3_files(bucket, prefix, pattern=pattern))
        return [str(p) for p in sorted(Path(config_dir).glob(f"{CONFIG_PREFIX}*.yaml"))]

    return _cached(f"list::{config_dir}", _produce)


def _load_yaml(uri: str) -> dict[str, Any]:
    def _produce() -> dict[str, Any]:
        return read_yaml_from_s3(uri) if is_s3_path(uri) else load_config(uri)

    return _cached(f"yaml::{uri}", _produce)


def find_config_path(config_dir: str, config_id: str) -> Optional[str]:
    for uri in list_config_files(config_dir):
        if _config_id(uri) == config_id:
            return uri
    return None


def list_config_summaries(config_dir: str) -> list[ConfigSummary]:
    summaries: list[ConfigSummary] = []
    for uri in list_config_files(config_dir):
        config_id = _config_id(uri)
        try:
            cfg = _load_yaml(uri)
        except Exception:  # a malformed config shouldn't blank the whole picker
            logger.exception("Failed to load config %s", uri)
            continue
        summaries.append(ConfigSummary(id=config_id, title=str(cfg.get("title", config_id))))
    return summaries


def load_raw_config(config_dir: str, config_id: str) -> Optional[dict[str, Any]]:
    uri = find_config_path(config_dir, config_id)
    if uri is None:
        return None
    cfg = _load_yaml(uri)
    # Stamp the id (the YAML has no id of its own — it comes from the filename).
    # The window cache in services/common.py keys on it; without this every
    # config collides on the same cache key and serves another game's rows.
    cfg["id"] = config_id
    return cfg


def build_config_detail(config_id: str, cfg: dict[str, Any]) -> ConfigDetail:
    stats_by_date = cfg.get("stats_by_date") or {}
    files = stats_by_date.get("files") or {}

    granularities = [g for g in _GRANULARITIES if files.get(g)]

    groups: list[MetricGroup] = []
    for i in (1, 2, 3):
        cols = stats_by_date.get(f"group{i}_columns")
        if cols:
            groups.append(
                MetricGroup(
                    id=f"group{i}",
                    label=str(stats_by_date.get(f"group{i}_label", f"Group {i}")),
                    metrics=[str(c) for c in cols],
                )
            )

    tabs: list[str] = []
    if stats_by_date:
        tabs += ["stats-by-date", "stats-by-group", "stats-deepdive"]
    if (cfg.get("stats_by_bet") or {}).get("files"):
        tabs.append("stats-by-bet")
    if cfg.get("weekly_report"):
        tabs.append("weekly-report")

    return ConfigDetail(
        id=config_id,
        title=str(cfg.get("title", config_id)),
        date_col=str(stats_by_date.get("date_col", "activity_date")),
        group_col=str(stats_by_date.get("group_col", "")),
        user_group_cols=[str(c) for c in (stats_by_date.get("user_group_cols") or [])],
        granularities=granularities,  # type: ignore[arg-type]
        groups=groups,
        tabs=tabs,
    )
