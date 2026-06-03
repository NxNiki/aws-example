"""Pydantic models for the /api/data/* surface.

These are the single source of truth for the data contract: FastAPI emits
them into the OpenAPI schema, from which the frontend's TypeScript client is
generated (see frontend/src/api/types.ts, hand-written in Phase 0).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Granularity = Literal["day", "week", "month"]


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


class Series(BaseModel):
    name: str
    y: list[Optional[float]]


class SeriesResponse(BaseModel):
    config: str
    granularity: Granularity
    date_col: str
    x: list[Optional[str]]  # ISO dates aligned with each series' y
    series: list[Series]
    missing: list[str]  # requested metrics not present in the source parquet
