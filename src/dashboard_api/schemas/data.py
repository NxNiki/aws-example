"""Pydantic models for the /api/data/* surface.

These are the single source of truth for the data contract: FastAPI emits
them into the OpenAPI schema, from which the frontend's TypeScript client is
generated (see frontend/src/api/types.ts, hand-written in Phase 0).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Granularity = Literal["day", "week", "month"]

# How a metric's values were produced, so the frontend knows whether to draw a
# CI band and how to combine across cohorts:
#   computed — a group-level DataMetrics aggregate (one value/date, no CI)
#   user     — a per-user (`user_*`) column: mean ± 95% bootstrap CI per date
#   raw      — any other column: mean per date, no CI
SeriesKind = Literal["computed", "user", "raw"]


class ConfigSummary(BaseModel):
    id: str
    title: str


class ConfigList(BaseModel):
    configs: list[ConfigSummary]


class MetricGroup(BaseModel):
    id: str
    label: str
    metrics: list[str]


class ConfigDetail(BaseModel):
    id: str
    title: str
    date_col: str
    group_col: str
    user_group_cols: list[str]
    # The user-group column the global lifecycle-group picker redefines (the
    # ETL's hard-coded new/beginner/old split). None → the config has no
    # lifecycle cohorts and the picker is hidden.
    lifecycle_col: Optional[str] = None
    # Numeric column the "range group" picker buckets (e.g. fish_value), plus
    # its display name and default groups; None → no such picker.
    range_group_col: Optional[str] = None
    range_group_name: Optional[str] = None
    range_group_defaults: list["RangeGroup"] = []
    # Config-declared bound ladder for the picker; empty → derive from data.
    range_group_values: list[float] = []
    # Derived period-total range picker ("Total bet"): virtual column name,
    # display name, and default groups per granularity (edges scale with the
    # period, so day/week/month each get their own). None → no picker.
    period_total_col: Optional[str] = None
    period_total_name: Optional[str] = None
    period_total_defaults: dict[str, list["RangeGroup"]] = {}
    granularities: list[Granularity]
    groups: list[MetricGroup]
    tabs: list[str]


class RangeGroup(BaseModel):
    """One user-defined value-range cohort over a range dimension: rows with
    min <= value <= max — both ends INCLUSIVE, ``max=None`` open-ended.

    Dashboard feature: the "Fish level" / "Bet level" / "Total bet" pickers
    next to Lifecycle groups. ``column`` says which range dimension the group
    addresses — the stored range column (``range_group_col``) when omitted
    (pre-multi-picker clients), or the derived period-total column
    (``period_total_col``, one total per user × period). Period-total ranges
    are HALF-OPEN [min, max) so adjacent tiers partition exactly; the stored
    range column keeps inclusive bounds (they are real ladder values). The
    label "all" is the no-filter sentinel; ranges may overlap.
    """

    label: str = Field(min_length=1)
    column: Optional[str] = None
    min: float = Field(ge=0)
    max: Optional[float] = None

    @model_validator(mode="after")
    def _check_range(self) -> "RangeGroup":
        if self.max is not None and self.max < self.min:
            raise ValueError(f"max ({self.max}) must be >= min ({self.min})")
        return self


class LifecycleGroup(BaseModel):
    """One user-defined lifecycle cohort: users whose activity falls in the
    HALF-OPEN period range [start, end) after their first bet — start
    inclusive, end exclusive, so adjacent groups sharing a boundary (0→3,
    3→7, 7→max) tile with no gap and no double-count. The period unit IS the
    request's granularity — days on daily data, calendar weeks on weekly data,
    calendar months on monthly data (week/month rows aggregate a whole period,
    so sub-period day ranges would slice by start weekday, not user age).

    Dashboard feature: the global "Lifecycle groups" picker above the tab bar.
    ``end=None`` means open-ended (start and later). The label "all" is a
    sentinel meaning no filter (the full population), mirroring group_values.
    Ranges may overlap (e.g. compare day 0→3 against day 0→7).
    """

    label: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> "LifecycleGroup":
        if self.end is not None and self.end <= self.start:
            raise ValueError(f"end ({self.end}) is exclusive and must be > start ({self.start})")
        return self


class SeriesRequest(BaseModel):
    config: str
    granularity: Granularity = "day"
    metrics: list[str] = Field(min_length=1)
    date_from: Optional[str] = None  # ISO date (YYYY-MM-DD), inclusive
    date_to: Optional[str] = None  # ISO date (YYYY-MM-DD), inclusive
    # Up to three comparable date windows (the global "Date groups" picker).
    # When set, supersedes date_from/date_to: one series is emitted per
    # metric × cohort × range, tagged with range_index/range_label, and the
    # frontend concatenates the ranges horizontally on the chart.
    ranges: Optional[list["DateRange"]] = None
    # Cohort selection: per user_group column, the values to keep. A column
    # absent here (or an empty list) means "all" — no filter on that column.
    # The response emits one series per requested metric × cohort combination.
    group_values: dict[str, list[str]] = Field(default_factory=dict)
    # When set, redefines the lifecycle user-group column (ConfigDetail.lifecycle_col)
    # with these day-since-first-bet ranges; it then IS the selection for that
    # column (group_values for it is ignored).
    lifecycle_groups: Optional[list[LifecycleGroup]] = None
    # Value-range cohorts over range_group_col (see RangeGroup); like
    # lifecycle_groups, when set it IS the selection for that dimension.
    range_groups: Optional[list[RangeGroup]] = None


class Series(BaseModel):
    metric: str
    cohort: str  # cohort label, e.g. "all", "new", "new|control"
    kind: SeriesKind
    additive: bool  # metric sums across cohorts (counts/totals) vs. ratio/rate
    x: list[str]  # ISO dates for this series (retention can extend the window)
    y: list[Optional[float]]
    lower: list[Optional[float]]  # CI lower bound (== y when kind != "user")
    upper: list[Optional[float]]  # CI upper bound (== y when kind != "user")
    # Which of the request's ranges produced this series (None for the
    # single-window date_from/date_to form).
    range_index: Optional[int] = None
    range_label: Optional[str] = None


class SeriesResponse(BaseModel):
    config: str
    granularity: Granularity
    date_col: str
    series: list[Series]
    missing: list[str]  # requested metrics with no data / missing source columns


class GroupValues(BaseModel):
    config: str
    granularity: Granularity
    # Distinct selectable values per user_group column, for the cohort pickers.
    values: dict[str, list[str]]
    # Subset of `values` present in the requested [start, end] date range; the
    # pickers gray out the rest. None when no range was requested.
    available: Optional[dict[str, list[str]]] = None


class DateBounds(BaseModel):
    config: str
    granularity: Granularity
    # Min/max ISO date present in the source data; the UI seeds its default
    # window (last 30 days) from `max`. Null when the data is empty.
    min: Optional[str]
    max: Optional[str]


# --- Stats by Group + Deep Dive (distribution / correlation) ---------------


# An inclusive date window. Both tabs compare up to 3 of these side by side.
class DateRange(BaseModel):
    start: Optional[str] = None  # ISO date (YYYY-MM-DD)
    end: Optional[str] = None


# Display-only value clipping: pin values outside [min, max] to the bound.
class ClipOpts(BaseModel):
    enable: bool = False
    min: Optional[float] = None
    max: Optional[float] = None


# Percentile filter: DROP samples below the min / above the max percentile
# (both in 0–100). Unlike clip (which pins values and keeps every sample),
# filtered samples are removed before any stats/binning. Applied before clip.
class FilterOpts(BaseModel):
    enable: bool = False
    min: Optional[float] = Field(default=None, ge=0, le=100)
    max: Optional[float] = Field(default=None, ge=0, le=100)


class GroupDistributionRequest(BaseModel):
    config: str
    granularity: Granularity = "day"
    metric: str
    ranges: list[DateRange] = Field(min_length=1)
    group_values: dict[str, list[str]] = Field(default_factory=dict)
    lifecycle_groups: Optional[list[LifecycleGroup]] = None
    range_groups: Optional[list[RangeGroup]] = None
    clip: ClipOpts = Field(default_factory=ClipOpts)
    filter: FilterOpts = Field(default_factory=FilterOpts)


# Per (cohort × date-range) distribution summary — enough to draw either a box
# (5-number summary) or a bar (mean ± bootstrap CI) without a refetch.
class GroupStat(BaseModel):
    cohort: str
    range_label: str
    range_index: int  # which of the request's ranges (0-based); drives dimming
    n: int
    min: Optional[float]
    q1: Optional[float]
    median: Optional[float]
    q3: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]


class GroupDistributionResponse(BaseModel):
    config: str
    granularity: Granularity
    metric: str
    stats: list[GroupStat]
    missing: bool  # the metric produced no data in any cohort/range


# --- Summary table (all metrics × all cohort/range columns in one grid) ------


# Per-metric display reshaping applied before stats/p-values: clip pins values
# to [min, max]; log applies a signed log1p. Independent per metric so a skewed
# metric can be tamed without touching the others.
class SummaryMetricOption(BaseModel):
    log: bool = False
    clip: ClipOpts = Field(default_factory=ClipOpts)


class SummaryTableRequest(BaseModel):
    config: str
    granularity: Granularity = "day"
    # The columns are the cross-product of selected cohorts × these ranges.
    ranges: list[DateRange] = Field(min_length=1)
    group_values: dict[str, list[str]] = Field(default_factory=dict)
    lifecycle_groups: Optional[list[LifecycleGroup]] = None
    range_groups: Optional[list[RangeGroup]] = None
    # Per-metric (keyed by metric name) clip + log; a metric absent here is left
    # as-is. Sent for every metric the client may show.
    metric_options: dict[str, SummaryMetricOption] = Field(default_factory=dict)
    # Compute the per-metric significance column (t-test for 2 columns, one-way
    # ANOVA for 3+). Off by default — it re-reads the per-row value arrays.
    pvalues: bool = False


# One column of the summary table: a single (cohort × date-range) combination.
class SummaryColumn(BaseModel):
    key: str  # stable id the frontend uses to track the reference column
    cohort: str
    range_index: int  # which of the request's ranges (0-based)
    range_label: str


# One metric's value summary in a single column. The frontend renders whichever
# of these stats the user has toggled on, each with its ±% vs the reference column.
class SummaryCell(BaseModel):
    n: int
    mean: Optional[float]
    median: Optional[float]
    q1: Optional[float]
    q3: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]


# One table row: a metric across every column (cells aligned to response.columns;
# None = no data for that cohort/range), plus the optional significance test.
class SummaryRow(BaseModel):
    metric: str
    group_id: str  # which metric group (group1/2/3) this row belongs to
    group_label: str
    cells: list[Optional[SummaryCell]]
    pvalue: Optional[float] = None
    test: Optional[str] = None  # "t-test" | "anova" | None
    missing: bool = False  # metric absent / produced no data in any column


class SummaryTableResponse(BaseModel):
    config: str
    granularity: Granularity
    columns: list[SummaryColumn]
    rows: list[SummaryRow]


DeepdivePanel = Literal["derived", "user"]  # group-level metrics vs raw user_* columns
DeepdiveMode = Literal["histogram", "heatmap", "scatter"]


class DeepdiveRequest(BaseModel):
    config: str
    granularity: Granularity = "day"
    panel: DeepdivePanel
    mode: DeepdiveMode
    metrics: list[str] = Field(min_length=1)
    ranges: list[DateRange] = Field(min_length=1)
    group_values: dict[str, list[str]] = Field(default_factory=dict)
    lifecycle_groups: Optional[list[LifecycleGroup]] = None
    range_groups: Optional[list[RangeGroup]] = None
    clip: ClipOpts = Field(default_factory=ClipOpts)
    filter: FilterOpts = Field(default_factory=FilterOpts)
    nbins: int = 50  # histogram only
    normalize: bool = False  # histogram only: density vs counts
    outliers_std: Optional[float] = None  # scatter only: drop rows beyond N std (None = off)


# One histogram (server-binned) for a (metric × cohort × range).
class HistogramSeries(BaseModel):
    metric: str
    cohort: str
    range_label: str
    range_index: int
    bin_edges: list[float]  # length = len(counts) + 1
    counts: list[float]


# One Pearson correlation matrix for a (cohort × range) across the metrics.
class CorrMatrix(BaseModel):
    cohort: str
    range_label: str
    range_index: int
    metrics: list[str]
    corr: list[list[Optional[float]]]


# Outlier-filtered, down-sampled points for one (cohort × range). `points` rows
# are aligned to `metrics` (point[k] = value of metrics[k]); the frontend builds
# the pair plot or scatter matrix from these.
class ScatterSeries(BaseModel):
    cohort: str
    range_label: str
    range_index: int
    metrics: list[str]
    points: list[list[float]]


class DeepdiveResponse(BaseModel):
    config: str
    granularity: Granularity
    panel: DeepdivePanel
    mode: DeepdiveMode
    histograms: list[HistogramSeries]
    heatmaps: list[CorrMatrix]
    scatters: list[ScatterSeries]
    missing: list[str]  # requested metrics with no data


class DeepdiveMetrics(BaseModel):
    config: str
    granularity: Granularity
    derived: list[str]  # DataMetrics group-level metrics
    user: list[str]  # raw user_* columns present in the data


class ViewList(BaseModel):
    # Names of saved dashboard view snapshots (the legacy "save/load config").
    views: list[str]


class ViewSaveResult(BaseModel):
    name: str  # sanitized name actually stored
    path: str  # s3:// path of the saved snapshot (shown to the user)
    views: list[str]
