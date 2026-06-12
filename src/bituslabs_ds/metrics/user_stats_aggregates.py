"""
Compute individual group-level daily metrics from one-row-per-user Parquet frames.

Each metric is a ``cached_property`` method — computed lazily on first access and cached
for all subsequent calls on the same instance. Each metric runs only the aggregation it
needs; no metric triggers the computation of any other unless it explicitly depends on it.

Usage::

    # Recommended dashboard flow:
    #   1. Load raw user-level Parquet into df (with extra days for retention).
    #   2. Filter by user_group BEFORE creating DataMetrics.
    #   3. Create DataMetrics on the filtered data — key_cols = [date_col, group_col].
    #   4. Call dm.plot_values(metric, df_group_raw) per group.

    # df_raw = lf.filter(date_range).collect()
    # df_filtered = apply_user_group_filter(df_raw, user_group)
    # dm = DataMetrics(df_filtered, start_dt=start, end_dt=end, key_cols=["activity_date", "ai_group"])
    # x, y, lo, hi = dm.plot_values("rtp", df_filtered.filter(pl.col("ai_group") == strat))

    # Access individual metric DataFrames:
    # dm["rtp"]                  # → pl.DataFrame [activity_date, ai_group, rtp]
    # dm["retention_rate_day1"]  # → pl.DataFrame [activity_date, ai_group, retention_rate_day1]
    # "rtp" in dm                # → True
    # dm.keys()                  # → sorted list of all known metric names

``df`` must cover ``[start_dt, end_dt + RETENTION_LOAD_EXTRA_DAYS]`` so that retention
follow-up dates are available when a retention metric is requested.

Public API (imported by game_stats_monitor):
    DataMetrics, ENRICH_PRODUCED_COLUMNS, ENRICH_USER_ROW_INPUT_COLUMNS,
    RETENTION_LOAD_EXTRA_DAYS, column_is_enrich_produced
"""

from __future__ import annotations

import inspect
import logging
import re
from datetime import date, datetime, timedelta
from functools import cached_property
from typing import ClassVar, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retention window constants
# ---------------------------------------------------------------------------

RETENTION_DAY_OFFSETS: Tuple[int, ...] = (1, 2, 3)
RETENTION_LOAD_EXTRA_DAYS: int = max(RETENTION_DAY_OFFSETS)

# ---------------------------------------------------------------------------
# Module-level pure utilities
# ---------------------------------------------------------------------------


def _to_datetime(d: Union[date, datetime]) -> datetime:
    return d if isinstance(d, datetime) else datetime.combine(d, datetime.min.time())


def _norm_date(x: Union[date, datetime]) -> date:
    return x.date() if isinstance(x, datetime) else x


def _bootstrap_ci(
    arr: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap percentile CI for the mean of ``arr``.

    Resamples in batches: a single-shot ``(n_boot, len(arr))`` matrix is
    ~800 MB for a 200k-element per-user array, and several land concurrently
    when a dashboard tab bootstraps its panels in parallel — this OOM-killed
    the dashboard-api task in production. Batching caps the transient at
    ~80 MB with identical statistics.
    """
    if len(arr) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng()
    batch = max(1, min(n_boot, 10_000_000 // len(arr)))
    boot_means = np.empty(n_boot)
    for i in range(0, n_boot, batch):
        k = min(batch, n_boot - i)
        boot_means[i : i + k] = rng.choice(arr, size=(k, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(boot_means, 100 * alpha / 2)), float(np.percentile(boot_means, 100 * (1 - alpha / 2)))


def _horizon_dates(d0: Union[date, datetime], granularity: str) -> Tuple[datetime, datetime, datetime]:
    """Three follow-up datetimes for a cohort starting at ``d0``."""
    t0 = _to_datetime(d0)
    g = (granularity or "day").lower()
    if g == "week":
        return t0 + timedelta(days=7), t0 + timedelta(days=14), t0 + timedelta(days=21)
    if g == "month":
        return t0 + timedelta(days=31), t0 + timedelta(days=62), t0 + timedelta(days=93)
    hs = tuple(t0 + timedelta(days=k) for k in RETENTION_DAY_OFFSETS)
    return hs  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# DataMetrics
# ---------------------------------------------------------------------------


class DataMetrics:
    """
    Lazy, per-metric accessor for group-level daily statistics.

    ``dm["rtp"]`` returns a ``pl.DataFrame`` with columns
    ``[date_col, user_group_col, group_col, metric_name]`` (``user_group_col`` is
    omitted when not configured), with one row per (date, [user_group,] group) tuple.
    Each metric is computed independently and cached on first access.
    """

    ACTIVE_USER_MIN_BETS: ClassVar[int] = 40  # minimum bets to be counted as an active user

    # Additive metrics — must be SUMMED when aggregating across user_groups.
    # These are totals and counts whose values add up correctly across segments.
    # Rate/ratio metrics (rtp, hit_rate, retention_rate, …) are NOT additive and
    # are excluded here; they are taken as-is (first) since DataMetrics already
    # computes them correctly at the group level.
    ADDITIVE_METRICS: ClassVar[FrozenSet[str]] = frozenset(
        {
            # user counts
            "num_active_users",
            "num_new_users",
            "num_active_user_0_rtp",
            "day0_num_users",
            "day1_num_users",
            "day2_num_users",
            "day3_num_users",
            # bet / payout / profit totals
            "total_num_bets",
            "total_num_bets_bg",
            "total_num_bets_fg",
            "total_bet",
            "total_bet_bg",
            "total_bet_fg",
            "total_payout",
            "total_payout_bg",
            "total_payout_fg",
            "total_num_bets_with_payout",
            "total_num_bets_bg_with_payout",
            "total_num_bets_fg_with_payout",
            "total_profit",
        }
    )

    METRICS: ClassVar[FrozenSet[str]] = frozenset(
        {
            # direct group-by aggregates
            "num_active_users",
            "num_new_users",
            "total_num_bets",
            "total_num_bets_bg",
            "total_num_bets_fg",
            "total_bet",
            "total_bet_bg",
            "total_payout",
            "total_payout_bg",
            "total_payout_fg",
            "total_bet_fg",
            "total_num_bets_with_payout",
            "total_num_bets_bg_with_payout",
            "total_num_bets_fg_with_payout",
            # derived
            "total_num_bets_per_user",
            "total_bet_per_user",
            "total_profit",
            "total_profit_per_user",
            "rtp",
            "rtp_bg",
            "rtp_fg",
            "fg_ratio",
            "hit_rate",
            "hit_rate_bg",
            "hit_rate_fg",
            "active_user_no_fg_ratio",
            "num_active_user_0_rtp",
            "active_user_0_rtp_ratio",
            "active_user_rtp_less_0_1_ratio",
            "active_user_rtp_less_0_2_ratio",
            "active_user_rtp_less_0_3_ratio",
            "active_user_rtp_less_0_4_ratio",
            "active_user_rtp_less_0_5_ratio",
            # RTP distribution
            "user_rtp_median",
            "user_rtp_ultilization_ratio",
            # retention
            "day0_num_users",
            "day1_num_users",
            "day2_num_users",
            "day3_num_users",
            "retention_rate_day1",
            "retention_rate_day3",
        }
    )

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        start_dt: datetime,
        end_dt: datetime,
        key_cols: List[str],
        granularity: str = "day",
    ) -> None:
        self._df = df
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.key_cols = key_cols
        self.granularity = granularity

    @property
    def date_col(self) -> str:
        """First element of ``key_cols`` — always the time dimension."""
        return self.key_cols[0]

    # ------------------------------------------------------------------
    # Schema-dependency introspection
    # ------------------------------------------------------------------

    _user_col_pattern: ClassVar = re.compile(r'pl\.col\(\s*"(user_[A-Za-z0-9_]+)"\s*\)')
    _self_method_pattern: ClassVar = re.compile(r"self\._([A-Za-z_][A-Za-z0-9_]*)")

    @classmethod
    def _source_user_col_deps(cls, attr_name: str, _seen: Optional[Set[str]] = None) -> Set[str]:
        """Pull ``user_*`` deps out of one ``self._<attr_name>`` body, recursing into helpers it calls.

        ``attr_name`` may be either a ``cached_property`` (e.g. ``rtp``) or a
        plain method (e.g. ``_count_at_horizon``). Internal helpers count
        too because retention metrics route through ``_cohorts`` /
        ``_presence_map`` rather than touching ``pl.col("user_*")`` directly.
        """
        if _seen is None:
            _seen = set()
        if attr_name in _seen:
            return set()
        _seen.add(attr_name)

        prop = getattr(cls, attr_name, None)
        func = getattr(prop, "func", None) or (prop if callable(prop) else None)
        if func is None:
            return set()
        try:
            src = inspect.getsource(func)
        except (OSError, TypeError):
            return set()

        deps = set(cls._user_col_pattern.findall(src))
        for ref in cls._self_method_pattern.findall(src):
            deps |= cls._source_user_col_deps(f"_{ref}", _seen)
        return deps

    @classmethod
    def metric_user_col_deps(cls, metric_name: str) -> Set[str]:
        """Return the ``user_*`` columns a metric needs (transitively).

        Used by the dashboard to filter which derived metrics are
        offerable for a given game's data — without it, a fishhunter
        config could expose slot-only metrics like ``active_user_no_fg_ratio``
        (depends on ``user_num_bets_fg``) and crash on access. We
        introspect each metric body for ``pl.col("user_*")`` references,
        chasing ``self._another_attr`` references through both
        ``cached_property`` metrics and internal helper methods. Pure
        regex is enough — metric bodies are simple aggregations.
        """
        return cls._source_user_col_deps(f"_{metric_name}")

    # ------------------------------------------------------------------
    # Dict-like interface
    # ------------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        """True if ``name`` is a computable metric *or* a raw column in the source data."""
        return isinstance(name, str) and (name in self.METRICS or name in self._df.columns)

    def keys(self) -> List[str]:
        return sorted(self.METRICS)

    def __getitem__(self, name: str) -> Optional[pl.DataFrame]:
        """Return a ``pl.DataFrame`` containing ``[*_key_cols, name]``, or ``None``.

        Works isomorphically for both computed and raw columns:

        - **Computed metric** (``name in METRICS``): returns a group-level DataFrame
          (one row per ``_key_cols`` combination) with the aggregated value.
        - **Raw column** (``name in self._df.columns``): returns
          ``self._df.select([*_key_cols, name])`` — the user-level rows as-is.
          The caller is responsible for any further aggregation (mean, bootstrap CI, etc.).
        - **Unknown** (neither): logs a warning and returns ``None``.
        """
        if name in self.METRICS:
            result: Optional[pl.DataFrame] = getattr(self, f"_{name}", None)
            if result is None or result.is_empty():
                logger.warning("DataMetrics[%r]: metric is defined but returned no data for this period", name)
                return None
            return result
        if name in self._df.columns:
            return self._df.select(self._key_cols + [name])
        logger.warning("DataMetrics[%r]: not a known metric and not a column in the source data", name)
        return None

    def extend(self) -> pl.DataFrame:
        """
        Return a **key-cols-level** DataFrame (one row per ``_key_cols`` combination)
        extended with every metric in ``METRICS``.

        Steps:
        1. Join each computable metric onto ``self._df`` (user-level rows).
        2. Collapse to ``_key_cols`` granularity so downstream plotting operates on
           group-level data, not duplicated per-user rows:
           - ``METRICS`` columns → ``first()``  (already identical within a group; preserves
             integer dtype for count metrics).
           - Other numeric columns → ``mean()``  (averages user-level values per group).
           - Other columns → ``first()``.
           - ``user_id`` is dropped (meaningless at group level).

        All warning / error logging is handled here; callers need no metric-level logic.
        """
        out = self._df
        for col in sorted(self.METRICS):
            if col in out.columns:
                continue
            metric_df = self[col]
            if metric_df is None:
                logger.warning(
                    "DataMetrics.extend: metric %r could not be computed "
                    "(no data for this period or missing source columns)",
                    col,
                )
                continue
            out = out.join(metric_df, on=self._key_cols, how="left")

        # Collapse user-level rows to key_cols granularity.
        schema = out.schema
        metric_cols_present = [c for c in self.METRICS if c in out.columns]
        other_cols = [
            c for c in out.columns if c not in self._key_cols and c not in metric_cols_present and c != "user_id"
        ]
        agg_exprs = [pl.col(c).first() for c in metric_cols_present]
        for c in other_cols:
            agg_exprs.append(pl.col(c).mean() if schema[c].is_numeric() else pl.col(c).first())

        return out.group_by(self._key_cols).agg(agg_exprs)

    def plot_values(
        self,
        metric: str,
        df_group_raw: pl.DataFrame,
        n_boot: int = 500,
    ) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ``(x_dates, y, y_lower, y_upper)`` ready for a time-series scatter plot.

        Uses ``self[metric]`` for both computed and raw columns (isomorphic API):

        - **Computed metrics** (in ``METRICS``): already group-level; returned as-is,
          no CI (one value per date, no distribution to sample from).
        - **Raw columns** (``user_*`` prefix): mean ± 95 % bootstrap CI across the
          per-user rows for each date.
        - **Other raw columns**: mean per date, no CI.

        ``df_group_raw`` is used only to identify which (date, group) keys to restrict
        to — it is not read for values directly.
        """
        date_col = self._key_cols[0]

        metric_df = self[metric]  # None only for truly unknown names (warning already logged)
        if metric_df is None:
            return [], np.array([]), np.array([]), np.array([])

        # Restrict to the (date, group) window of the caller's filtered slice.
        key_filter = df_group_raw.select(self._key_cols).unique()
        metric_df = metric_df.join(key_filter, on=self._key_cols, how="inner").sort(date_col)
        if metric_df.is_empty():
            return [], np.array([]), np.array([]), np.array([])

        if metric in self.METRICS:
            # Group-level: one value per date, no CI needed.
            y = metric_df.get_column(metric).to_numpy().astype(float)
            x = metric_df.get_column(date_col).to_list()
            return x, y, y.copy(), y.copy()

        # Raw column: aggregate per date (with optional bootstrap for user_* columns).
        dates = metric_df.get_column(date_col).unique().sort().to_list()
        do_bootstrap = metric.startswith("user_")
        means: List[float] = []
        lowers: List[float] = []
        uppers: List[float] = []
        for d in dates:
            arr = metric_df.filter(pl.col(date_col) == d).get_column(metric).drop_nulls().to_numpy().astype(float)
            if len(arr) == 0:
                means.append(float("nan"))
                lowers.append(float("nan"))
                uppers.append(float("nan"))
            else:
                m_val = float(arr.mean())
                means.append(m_val)
                if do_bootstrap and len(arr) >= 2:
                    lo, hi = _bootstrap_ci(arr, n_boot=n_boot)
                    lowers.append(lo if not np.isnan(lo) else m_val)
                    uppers.append(hi if not np.isnan(hi) else m_val)
                else:
                    lowers.append(m_val)
                    uppers.append(m_val)

        y = np.array(means)
        return dates, y, np.array(lowers), np.array(uppers)

    # ------------------------------------------------------------------
    # Shared building blocks (each cached; computing a metric never
    # re-triggers an already-computed building block)
    # ------------------------------------------------------------------

    @cached_property
    def _key_cols(self) -> List[str]:
        """
        ``key_cols`` filtered to columns that are actually present in ``self._df``.
        This allows graceful degradation when, e.g., ``user_group`` has not yet been
        backfilled into older Parquet files.
        """
        return [c for c in self.key_cols if c in self._df.columns]

    @cached_property
    def _user_rows(self) -> pl.DataFrame:
        """User rows in [start_dt, end_dt] for all user groups."""
        in_window = (pl.col(self.date_col) >= self.start_dt) & (pl.col(self.date_col) <= self.end_dt)
        rows = self._df.filter(in_window)
        self._warn_on_missing_source_values(rows)
        return rows

    @staticmethod
    def _warn_on_missing_source_values(rows: pl.DataFrame) -> None:
        """Warn when metric source columns contain null / NaN values.

        These usually indicate stale or partially-written upstream ETL output (e.g. an
        ETL rerun that appended onto existing parquet instead of overwriting).  Such
        nulls silently flow into derived metrics and produce missing cells in heatmaps
        and gaps in line plots.  Re-run the upstream ETL job with ``--overwrite`` to
        regenerate the source parquet cleanly.
        """
        if rows.is_empty():
            return
        cols_to_check = [c for c in ENRICH_USER_ROW_INPUT_COLUMNS if c in rows.columns]
        if not cols_to_check:
            return

        null_counts = rows.select(cols_to_check).null_count().row(0, named=True)
        null_cols = {c: int(n) for c, n in null_counts.items() if n}

        nan_cols: Dict[str, int] = {}
        schema = rows.schema
        for c in cols_to_check:
            if schema[c].is_float():
                n = rows.select(pl.col(c).is_nan().sum()).item()
                if n:
                    nan_cols[c] = int(n)

        if not null_cols and not nan_cols:
            return

        parts: List[str] = []
        if null_cols:
            parts.append(f"null counts: {dict(sorted(null_cols.items()))}")
        if nan_cols:
            parts.append(f"NaN counts: {dict(sorted(nan_cols.items()))}")
        logger.warning(
            "DataMetrics source data has missing values (%s). This typically indicates a "
            "stale or partially-written upstream ETL output and can produce empty cells in "
            "downstream plots / heatmaps. Re-run the upstream ETL job with --overwrite to "
            "regenerate the source parquet cleanly.",
            "; ".join(parts),
        )

    @cached_property
    def _presence_map(self) -> Dict[Tuple, Set]:
        """
        ``(*key_cols_values,) → set of user_ids`` across the FULL df (including follow-up dates).
        The date position in the key is normalised to ``date`` (not ``datetime``).
        """
        date_col = self._key_cols[0]
        presence: Dict[Tuple, Set] = {}
        for row in self._df.select(["user_id"] + self._key_cols).unique().iter_rows(named=True):
            key = tuple(_norm_date(row[c]) if c == date_col else row[c] for c in self._key_cols)
            presence.setdefault(key, set()).add(row["user_id"])
        return presence

    @cached_property
    def _cohorts(self) -> List[Tuple]:
        """
        ``[(key_row, d0, n0, cohort_set), ...]`` for each unique key-column combo in user_rows.
        ``key_row`` is a ``dict`` mapping each ``_key_col`` to its value for that combo.
        Cohort membership is ALL distinct users on each day (no bet threshold) so that
        ``day0_num_users`` tracks the full retention cohort.
        Shared by all retention metrics.

        Uses a single ``group_by`` to collect user_id sets — no per-row DataFrame filtering.
        """
        ur = self._user_rows
        if ur.is_empty():
            return []
        date_col = self._key_cols[0]
        grouped = ur.group_by(self._key_cols).agg(pl.col("user_id").unique().alias("_ids"))
        result = []
        for row in grouped.iter_rows(named=True):
            ids: List = list(row["_ids"])
            if ids:
                combo = {c: row[c] for c in self._key_cols}
                result.append((combo, row[date_col], len(ids), set(ids)))
        return result

    def _count_at_horizon(self, horizon_idx: int) -> Optional[pl.DataFrame]:
        """
        For each cohort, count how many users appear at the ``horizon_idx``-th follow-up date.
        Returns a DataFrame with ``_key_cols + ["_n"]`` or ``None`` when no cohorts exist.
        Not cached itself — but uses cached ``_cohorts`` and ``_presence_map``.
        """
        cohorts = self._cohorts
        if not cohorts:
            return None
        date_col = self._key_cols[0]
        presence = self._presence_map
        rows = []
        for combo, d0, _n0, cohort in cohorts:
            h = _horizon_dates(d0, self.granularity)[horizon_idx]
            h_key = tuple(_norm_date(h) if c == date_col else combo[c] for c in self._key_cols)
            n = len(cohort & presence.get(h_key, set()))
            rows.append({**combo, "_n": n})
        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # Direct group-by aggregation metrics
    # ------------------------------------------------------------------

    @cached_property
    def _num_active_users(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows.filter(pl.col("user_num_bets") >= self.ACTIVE_USER_MIN_BETS)
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(pl.col("user_id").n_unique().alias("num_active_users"))

    @cached_property
    def _num_new_users(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        if "user_group" not in ur.columns:
            return None
        return (
            ur.filter(pl.col("user_group") == "new")
            .group_by(self._key_cols)
            .agg(pl.col("user_id").n_unique().alias("num_new_users"))
        )

    @cached_property
    def _total_num_bets(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(pl.col("user_num_bets").sum().alias("total_num_bets"))

    @cached_property
    def _total_num_bets_bg(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(pl.col("user_num_bets_bg").sum().alias("total_num_bets_bg"))

    @cached_property
    def _total_num_bets_fg(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(pl.col("user_num_bets_fg").sum().alias("total_num_bets_fg"))

    @cached_property
    def _total_bet(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(pl.col("user_total_bet").sum().alias("total_bet"))

    @cached_property
    def _total_bet_bg(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(pl.col("user_total_bet_bg").sum().alias("total_bet_bg"))

    @cached_property
    def _total_payout(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(pl.col("user_total_payout").sum().alias("total_payout"))

    @cached_property
    def _total_payout_bg(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(pl.col("user_total_payout_bg").sum().alias("total_payout_bg"))

    @cached_property
    def _total_payout_fg(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(pl.col("user_total_payout_fg").sum().alias("total_payout_fg"))

    @cached_property
    def _total_bet_fg(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(pl.col("user_total_bet_fg").sum().alias("total_bet_fg"))

    @cached_property
    def _total_num_bets_with_payout(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(
            pl.col("user_num_bets_with_payout").sum().alias("total_num_bets_with_payout")
        )

    @cached_property
    def _total_num_bets_bg_with_payout(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(
            pl.col("user_num_bets_bg_with_payout").sum().alias("total_num_bets_bg_with_payout")
        )

    @cached_property
    def _total_num_bets_fg_with_payout(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(
            pl.col("user_num_bets_fg_with_payout").sum().alias("total_num_bets_fg_with_payout")
        )

    # ------------------------------------------------------------------
    # Derived metrics (each runs its own minimal agg)
    # ------------------------------------------------------------------

    @cached_property
    def _total_num_bets_per_user(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                pl.col("user_num_bets").sum().alias("_bets"),
                pl.col("user_id").n_unique().alias("_n"),
            )
            .with_columns((pl.col("_bets") / pl.col("_n").replace(0, None)).alias("total_num_bets_per_user"))
            .select(self._key_cols + ["total_num_bets_per_user"])
        )

    @cached_property
    def _total_bet_per_user(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                pl.col("user_total_bet").sum().alias("_bet"),
                pl.col("user_id").n_unique().alias("_n"),
            )
            .with_columns((pl.col("_bet") / pl.col("_n").replace(0, None)).alias("total_bet_per_user"))
            .select(self._key_cols + ["total_bet_per_user"])
        )

    @cached_property
    def _total_profit(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                pl.col("user_total_payout").sum().alias("_payout"),
                pl.col("user_total_bet").sum().alias("_bet"),
            )
            .with_columns((pl.col("_payout") - pl.col("_bet")).alias("total_profit"))
            .select(self._key_cols + ["total_profit"])
        )

    @cached_property
    def _total_profit_per_user(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                pl.col("user_total_payout").sum().alias("_payout"),
                pl.col("user_total_bet").sum().alias("_bet"),
                pl.col("user_id").n_unique().alias("_n"),
            )
            .with_columns(
                ((pl.col("_payout") - pl.col("_bet")) / pl.col("_n").replace(0, None)).alias("total_profit_per_user")
            )
            .select(self._key_cols + ["total_profit_per_user"])
        )

    @cached_property
    def _rtp(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                pl.col("user_total_payout").sum().alias("_payout"),
                pl.col("user_total_bet").sum().alias("_bet"),
            )
            .with_columns((pl.col("_payout") / pl.col("_bet").replace(0, None)).alias("rtp"))
            .select(self._key_cols + ["rtp"])
        )

    @cached_property
    def _rtp_bg(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                pl.col("user_total_payout_bg").sum().alias("_payout_bg"),
                pl.col("user_total_bet").sum().alias("_bet"),
            )
            .with_columns((pl.col("_payout_bg") / pl.col("_bet").replace(0, None)).alias("rtp_bg"))
            .select(self._key_cols + ["rtp_bg"])
        )

    @cached_property
    def _rtp_fg(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                pl.col("user_total_payout_fg").sum().alias("_payout_fg"),
                pl.col("user_total_bet_fg").sum().alias("_bet_fg"),
            )
            .with_columns((pl.col("_payout_fg") / pl.col("_bet_fg").replace(0, None)).alias("rtp_fg"))
            .select(self._key_cols + ["rtp_fg"])
        )

    @cached_property
    def _fg_ratio(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                pl.col("user_num_bets_fg").sum().alias("_fg"),
                pl.col("user_num_bets").sum().alias("_total"),
            )
            .with_columns((pl.col("_fg") / pl.col("_total").replace(0, None)).alias("fg_ratio"))
            .select(self._key_cols + ["fg_ratio"])
        )

    @cached_property
    def _hit_rate(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                pl.col("user_num_bets_with_payout").sum().alias("_with_payout"),
                pl.col("user_num_bets").sum().alias("_total"),
            )
            .with_columns((pl.col("_with_payout") / pl.col("_total").replace(0, None)).alias("hit_rate"))
            .select(self._key_cols + ["hit_rate"])
        )

    @cached_property
    def _hit_rate_bg(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                pl.col("user_num_bets_bg_with_payout").sum().alias("_with_payout"),
                pl.col("user_num_bets_bg").sum().alias("_total"),
            )
            .with_columns((pl.col("_with_payout") / pl.col("_total").replace(0, None)).alias("hit_rate_bg"))
            .select(self._key_cols + ["hit_rate_bg"])
        )

    @cached_property
    def _hit_rate_fg(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                pl.col("user_num_bets_fg_with_payout").sum().alias("_with_payout"),
                pl.col("user_num_bets_fg").sum().alias("_total"),
            )
            .with_columns((pl.col("_with_payout") / pl.col("_total").replace(0, None)).alias("hit_rate_fg"))
            .select(self._key_cols + ["hit_rate_fg"])
        )

    @cached_property
    def _active_user_no_fg_ratio(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                (pl.col("user_num_bets_fg") == 0).cast(pl.Int32).sum().alias("_no_fg"),
                pl.col("user_id").n_unique().alias("_n"),
            )
            .with_columns(
                (pl.col("_no_fg").cast(pl.Float64) / pl.col("_n").replace(0, None)).alias("active_user_no_fg_ratio")
            )
            .select(self._key_cols + ["active_user_no_fg_ratio"])
        )

    @cached_property
    def _num_active_user_0_rtp(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return ur.group_by(self._key_cols).agg(
            (pl.col("user_total_payout") == 0).cast(pl.Int64).sum().alias("num_active_user_0_rtp")
        )

    @cached_property
    def _active_user_0_rtp_ratio(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                (pl.col("user_total_payout") == 0).cast(pl.Int32).sum().alias("_count"),
                pl.col("user_id").n_unique().alias("_n"),
            )
            .with_columns(
                (pl.col("_count").cast(pl.Float64) / pl.col("_n").replace(0, None)).alias("active_user_0_rtp_ratio")
            )
            .select(self._key_cols + ["active_user_0_rtp_ratio"])
        )

    @cached_property
    def _active_user_rtp_less_0_1_ratio(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                ((pl.col("user_total_bet") > 0) & (pl.col("user_rtp") < 0.1)).sum().alias("_count"),
                pl.col("user_id").n_unique().alias("_n"),
            )
            .with_columns(
                (pl.col("_count").cast(pl.Float64) / pl.col("_n").replace(0, None)).alias(
                    "active_user_rtp_less_0_1_ratio"
                )
            )
            .select(self._key_cols + ["active_user_rtp_less_0_1_ratio"])
        )

    @cached_property
    def _active_user_rtp_less_0_2_ratio(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                ((pl.col("user_total_bet") > 0) & (pl.col("user_rtp") < 0.2)).sum().alias("_count"),
                pl.col("user_id").n_unique().alias("_n"),
            )
            .with_columns(
                (pl.col("_count").cast(pl.Float64) / pl.col("_n").replace(0, None)).alias(
                    "active_user_rtp_less_0_2_ratio"
                )
            )
            .select(self._key_cols + ["active_user_rtp_less_0_2_ratio"])
        )

    @cached_property
    def _active_user_rtp_less_0_3_ratio(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                ((pl.col("user_total_bet") > 0) & (pl.col("user_rtp") < 0.3)).sum().alias("_count"),
                pl.col("user_id").n_unique().alias("_n"),
            )
            .with_columns(
                (pl.col("_count").cast(pl.Float64) / pl.col("_n").replace(0, None)).alias(
                    "active_user_rtp_less_0_3_ratio"
                )
            )
            .select(self._key_cols + ["active_user_rtp_less_0_3_ratio"])
        )

    @cached_property
    def _active_user_rtp_less_0_4_ratio(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                ((pl.col("user_total_bet") > 0) & (pl.col("user_rtp") < 0.4)).sum().alias("_count"),
                pl.col("user_id").n_unique().alias("_n"),
            )
            .with_columns(
                (pl.col("_count").cast(pl.Float64) / pl.col("_n").replace(0, None)).alias(
                    "active_user_rtp_less_0_4_ratio"
                )
            )
            .select(self._key_cols + ["active_user_rtp_less_0_4_ratio"])
        )

    @cached_property
    def _active_user_rtp_less_0_5_ratio(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.group_by(self._key_cols)
            .agg(
                ((pl.col("user_total_bet") > 0) & (pl.col("user_rtp") < 0.5)).sum().alias("_count"),
                pl.col("user_id").n_unique().alias("_n"),
            )
            .with_columns(
                (pl.col("_count").cast(pl.Float64) / pl.col("_n").replace(0, None)).alias(
                    "active_user_rtp_less_0_5_ratio"
                )
            )
            .select(self._key_cols + ["active_user_rtp_less_0_5_ratio"])
        )

    # ------------------------------------------------------------------
    # RTP distribution metrics
    # ------------------------------------------------------------------

    @cached_property
    def _user_rtp_median(self) -> Optional[pl.DataFrame]:
        ur = self._user_rows
        if ur.is_empty():
            return None
        return (
            ur.filter(pl.col("user_total_bet") > 0)
            .with_columns((pl.col("user_total_payout") / pl.col("user_total_bet")).alias("_rtp_u"))
            .group_by(self._key_cols)
            .agg(pl.col("_rtp_u").median().alias("user_rtp_median"))
        )

    @cached_property
    def _user_rtp_ultilization_ratio(self) -> Optional[pl.DataFrame]:
        """user_rtp_median / rtp — uses both cached properties above."""
        med = self._user_rtp_median
        rtp = self._rtp
        if med is None or rtp is None:
            return None
        return (
            med.join(rtp, on=self._key_cols, how="left")
            .with_columns(
                (pl.col("user_rtp_median") / pl.col("rtp").replace(0, None)).alias("user_rtp_ultilization_ratio")
            )
            .select(self._key_cols + ["user_rtp_ultilization_ratio"])
        )

    # ------------------------------------------------------------------
    # Retention metrics
    # (presence_map + cohorts are shared; each day offset is independent)
    # ------------------------------------------------------------------

    @cached_property
    def _day0_num_users(self) -> Optional[pl.DataFrame]:
        """
        All distinct users per key-cols group on day 0 — the retention cohort size.
        No bet threshold: every user who appears on a date is part of that day's cohort.
        (Compare with ``num_active_users`` which requires ``user_num_bets >= ACTIVE_USER_MIN_BETS``.)
        """
        cohorts = self._cohorts
        if not cohorts:
            return None
        rows = [{**combo, "day0_num_users": n0} for combo, _d0, n0, _ in cohorts]
        return pl.DataFrame(rows)

    @cached_property
    def _day1_num_users(self) -> Optional[pl.DataFrame]:
        frame = self._count_at_horizon(0)
        if frame is None:
            return None
        return frame.rename({"_n": "day1_num_users"})

    @cached_property
    def _day2_num_users(self) -> Optional[pl.DataFrame]:
        frame = self._count_at_horizon(1)
        if frame is None:
            return None
        return frame.rename({"_n": "day2_num_users"})

    @cached_property
    def _day3_num_users(self) -> Optional[pl.DataFrame]:
        frame = self._count_at_horizon(2)
        if frame is None:
            return None
        return frame.rename({"_n": "day3_num_users"})

    @cached_property
    def _retention_rate_day1(self) -> Optional[pl.DataFrame]:
        day0 = self._day0_num_users
        day1 = self._day1_num_users
        if day0 is None or day1 is None:
            return None
        return (
            day0.join(day1, on=self._key_cols, how="left")
            .with_columns(
                (pl.col("day1_num_users") / pl.col("day0_num_users").replace(0, None)).alias("retention_rate_day1")
            )
            .select(self._key_cols + ["retention_rate_day1"])
        )

    @cached_property
    def _retention_rate_day3(self) -> Optional[pl.DataFrame]:
        day0 = self._day0_num_users
        day3 = self._day3_num_users
        if day0 is None or day3 is None:
            return None
        return (
            day0.join(day3, on=self._key_cols, how="left")
            .with_columns(
                (pl.col("day3_num_users") / pl.col("day0_num_users").replace(0, None)).alias("retention_rate_day3")
            )
            .select(self._key_cols + ["retention_rate_day3"])
        )


# ---------------------------------------------------------------------------
# Module-level aliases (for backward compat with game_stats_monitor imports)
# ---------------------------------------------------------------------------

ENRICH_PRODUCED_COLUMNS: FrozenSet[str] = DataMetrics.METRICS

ENRICH_USER_ROW_INPUT_COLUMNS: FrozenSet[str] = frozenset(
    {
        "user_id",
        "user_num_bets",
        "user_num_bets_bg",
        "user_num_bets_fg",
        "user_total_bet",
        "user_total_bet_bg",
        "user_avg_bet_amount",
        "user_total_payout",
        "user_total_payout_bg",
        "user_total_payout_fg",
        "user_total_bet_fg",
        "user_num_bets_with_payout",
        "user_num_bets_bg_with_payout",
        "user_num_bets_fg_with_payout",
        "user_rtp",
    }
)


def column_is_enrich_produced(name: str) -> bool:
    return name in DataMetrics.METRICS
