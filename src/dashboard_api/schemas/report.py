"""Pydantic models for the Report Spec and the /api/report/* surface.

The Report Spec is "the data that can recover the report"
(docs/frontend_redesign.md §6): figures are stored as declarative *recipes*
(which config/tab/panel/metrics/windows to fetch), NOT rendered images or
frozen figures — the Report tab re-fetches from /api/data/* and re-renders,
which is what makes "regenerate the report with a new period" possible.
Specs are persisted as named JSON files in S3 (versioned bucket).
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from dashboard_api.schemas.data import ClipOpts, DateRange, Granularity, SummaryMetricOption

ReportLanguage = Literal["en", "zh-Hans", "zh-Hant"]


# ── Figure recipes (one per source tab kind) ───────────────────────────────


class DateFigureSource(BaseModel):
    kind: Literal["stats-by-date"] = "stats-by-date"
    config: str
    granularity: Granularity = "day"
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    cohort_selection: dict[str, list[str]] = Field(default_factory=dict)
    panel_id: str
    left: list[str] = Field(default_factory=list)
    right: list[str] = Field(default_factory=list)
    log: bool = False
    threshold: float = 10


class GroupFigureSource(BaseModel):
    kind: Literal["stats-by-group"] = "stats-by-group"
    config: str
    granularity: Granularity = "day"
    ranges: list[DateRange] = Field(default_factory=list)
    cohort_selection: dict[str, list[str]] = Field(default_factory=dict)
    panel_id: str
    metric: str
    mode: Literal["box", "bar"] = "bar"
    clip: ClipOpts = Field(default_factory=ClipOpts)


class DeepdiveFigureSource(BaseModel):
    kind: Literal["stats-deepdive"] = "stats-deepdive"
    config: str
    granularity: Granularity = "day"
    ranges: list[DateRange] = Field(default_factory=list)
    cohort_selection: dict[str, list[str]] = Field(default_factory=dict)
    panel: Literal["derived", "user"]
    mode: Literal["histogram", "heatmap", "scatter"] = "histogram"
    metrics: list[str] = Field(default_factory=list)
    nbins: int = 50
    normalize: bool = False
    outliers_std: Optional[float] = None
    scatter_log_x: bool = False
    scatter_log_y: bool = False
    log_y: bool = False
    clip: ClipOpts = Field(default_factory=ClipOpts)


class SummaryTableFigureSource(BaseModel):
    kind: Literal["summary-table"] = "summary-table"
    config: str
    granularity: Granularity = "day"
    ranges: list[DateRange] = Field(default_factory=list)
    cohort_selection: dict[str, list[str]] = Field(default_factory=dict)
    # Per metric-group (group id → selected metrics); display filter on the grid.
    metrics: dict[str, list[str]] = Field(default_factory=dict)
    stats: list[str] = Field(default_factory=lambda: ["mean"])
    reference_key: Optional[str] = None
    metric_options: dict[str, SummaryMetricOption] = Field(default_factory=dict)
    pvalues: bool = False


FigureSource = Union[DateFigureSource, GroupFigureSource, DeepdiveFigureSource, SummaryTableFigureSource]


class ReportFigure(BaseModel):
    id: str
    title: str = ""
    description: str = ""  # prose; may contain /prompt lines consumed on (re)generation
    # When true, the figure's date window follows the spec-level period on
    # regenerate (the "update the report with new data" lever).
    inherit_period: bool = True
    source: FigureSource = Field(discriminator="kind")


class ReportReference(BaseModel):
    id: str
    url: str
    title: Optional[str] = None
    text: Optional[str] = None
    status: Optional[str] = None  # ok | error | external
    error: Optional[str] = None


class ReportSpec(BaseModel):
    version: int = 1
    id: str = ""
    title: str = ""
    language: ReportLanguage = "en"
    period: Optional[DateRange] = None  # spec-level window; inherited by figures
    # Name of the saved dashboard view this report was authored from (one view
    # can be linked by many reports). Loading the report also restores it.
    # Figures still embed their own recipes — the link is context, not a
    # rendering dependency.
    view: Optional[str] = None
    figures: list[ReportFigure] = Field(default_factory=list)
    references: list[ReportReference] = Field(default_factory=list)
    summary: str = ""
    confluence_url: str = ""


# ── Endpoint payloads ───────────────────────────────────────────────────────


class ReportSpecList(BaseModel):
    specs: list[str]


class ReportSpecSaveResult(BaseModel):
    name: str
    path: str  # s3:// path written (surfaced in the UI toast)
    specs: list[str]


class ReferencesRequest(BaseModel):
    urls: list[str]


class ReferencesResponse(BaseModel):
    references: list[dict[str, str]]  # {url, title, text, error?} from load_references


# Description / summary proxies mirror the ai_agent payloads (opaque dicts —
# the agent service owns the contract; dashboard_api just forwards).
class GenerateProxyResponse(BaseModel):
    text: str
    status: str
    message: str
    elapsed_ms: int


class ExportFigure(BaseModel):
    title: str = ""
    description: str = ""
    png_base64: str  # client-rendered ECharts PNG (data-URL body, no prefix)


class ExportRequest(BaseModel):
    confluence_url: str
    summary: str = ""
    references: list[ReportReference] = Field(default_factory=list)
    figures: list[ExportFigure] = Field(min_length=1, max_length=20)


class ExportResponse(BaseModel):
    page_id: str
    figures_uploaded: int
    page_version: Optional[int]


# Re-exported for the OpenAPI client; FastAPI needs them referenced.
__all__ = [
    "ReportSpec",
    "ReportFigure",
    "FigureSource",
    "DateFigureSource",
    "GroupFigureSource",
    "DeepdiveFigureSource",
    "SummaryTableFigureSource",
    "ReportReference",
    "ReportSpecList",
    "ReportSpecSaveResult",
    "ReferencesRequest",
    "ReferencesResponse",
    "GenerateProxyResponse",
    "ExportFigure",
    "ExportRequest",
    "ExportResponse",
    "ReportLanguage",
]
