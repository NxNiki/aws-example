"""Discover and parse the dashboard_config-*.yaml files into API models.

Wraps `bituslabs_ds.dashboard_utils.load_config` so the React frontend gets
the same per-game config (metrics, groups, granularities, tabs) the legacy
Dash app reads, without depending on the Dash app.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from bituslabs_ds.dashboard_utils import load_config
from dashboard_api.schemas.data import ConfigDetail, ConfigSummary, MetricGroup

logger = logging.getLogger(__name__)

CONFIG_PREFIX = "dashboard_config-"
_GRANULARITIES = ("day", "week", "month")


def _config_id(path: Path) -> str:
    name = path.stem
    return name[len(CONFIG_PREFIX) :] if name.startswith(CONFIG_PREFIX) else name


def list_config_files(config_dir: str) -> list[Path]:
    return sorted(Path(config_dir).glob(f"{CONFIG_PREFIX}*.yaml"))


def find_config_path(config_dir: str, config_id: str) -> Optional[Path]:
    for path in list_config_files(config_dir):
        if _config_id(path) == config_id:
            return path
    return None


def list_config_summaries(config_dir: str) -> list[ConfigSummary]:
    summaries: list[ConfigSummary] = []
    for path in list_config_files(config_dir):
        try:
            cfg = load_config(str(path))
        except Exception:  # a malformed config shouldn't blank the whole picker
            logger.exception("Failed to load config %s", path)
            continue
        summaries.append(ConfigSummary(id=_config_id(path), title=str(cfg.get("title", _config_id(path)))))
    return summaries


def load_raw_config(config_dir: str, config_id: str) -> Optional[dict[str, Any]]:
    path = find_config_path(config_dir, config_id)
    if path is None:
        return None
    cfg = load_config(str(path))
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
