"""Pydantic models for the /api/data/* surface.

These are the single source of truth for the data contract: FastAPI emits
them into the OpenAPI schema, from which the frontend's TypeScript client is
generated (see frontend/src/api/types.ts, hand-written in Phase 0).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

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
    granularities: list[Granularity]
    groups: list[MetricGroup]
    tabs: list[str]


class SeriesRequest(BaseModel):
    config: str
    granularity: Granularity = "day"
    metrics: list[str] = Field(min_length=1)
    date_from: Optional[str] = None  # ISO date (YYYY-MM-DD), inclusive
    date_to: Optional[str] = None  # ISO date (YYYY-MM-DD), inclusive
    # Cohort selection: per user_group column, the values to keep. A column
    # absent here (or an empty list) means "all" — no filter on that column.
    # The response emits one series per requested metric × cohort combination.
    group_values: dict[str, list[str]] = Field(default_factory=dict)


class Series(BaseModel):
    metric: str
    cohort: str  # cohort label, e.g. "all", "new", "new|control"
    kind: SeriesKind
    additive: bool  # metric sums across cohorts (counts/totals) vs. ratio/rate
    x: list[str]  # ISO dates for this series (retention can extend the window)
    y: list[Optional[float]]
    lower: list[Optional[float]]  # CI lower bound (== y when kind != "user")
    upper: list[Optional[float]]  # CI upper bound (== y when kind != "user")


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


class GroupDistributionRequest(BaseModel):
    config: str
    granularity: Granularity = "day"
    metric: str
    ranges: list[DateRange] = Field(min_length=1)
    group_values: dict[str, list[str]] = Field(default_factory=dict)
    clip: ClipOpts = Field(default_factory=ClipOpts)


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


DeepdivePanel = Literal["derived", "user"]  # group-level metrics vs raw user_* columns
DeepdiveMode = Literal["histogram", "heatmap"]


class DeepdiveRequest(BaseModel):
    config: str
    granularity: Granularity = "day"
    panel: DeepdivePanel
    mode: DeepdiveMode
    metrics: list[str] = Field(min_length=1)
    ranges: list[DateRange] = Field(min_length=1)
    group_values: dict[str, list[str]] = Field(default_factory=dict)
    clip: ClipOpts = Field(default_factory=ClipOpts)
    nbins: int = 50  # histogram only
    normalize: bool = False  # histogram only: density vs counts


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


class DeepdiveResponse(BaseModel):
    config: str
    granularity: Granularity
    panel: DeepdivePanel
    mode: DeepdiveMode
    histograms: list[HistogramSeries]
    heatmaps: list[CorrMatrix]
    missing: list[str]  # requested metrics with no data


class DeepdiveMetrics(BaseModel):
    config: str
    granularity: Granularity
    derived: list[str]  # DataMetrics group-level metrics
    user: list[str]  # raw user_* columns present in the data
