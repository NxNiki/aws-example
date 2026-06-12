"""Report API: spec persistence, references, LLM generation proxy, Confluence export.

The Report tab's backend (docs/frontend_redesign.md §6). Specs are declarative
recipes saved to S3; description/summary generation proxies to the ai_agent
service (the architecture's one cross-service hop); export receives
client-rendered PNGs and writes the Confluence page.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from dashboard_api.schemas.report import (
    ExportRequest,
    ExportResponse,
    GenerateProxyResponse,
    ReferencesRequest,
    ReferencesResponse,
    ReportSpec,
    ReportSpecList,
    ReportSpecSaveResult,
)
from dashboard_api.services.report import ReportError, list_specs, load_spec, proxy_generate, save_spec
from dashboard_api.services.report_export import ExportError, export_report_pngs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/report", tags=["report"])


@router.get("/specs", response_model=ReportSpecList)
def get_specs() -> ReportSpecList:
    """API endpoint: GET /api/report/specs — list saved report specs."""
    try:
        return ReportSpecList(specs=list_specs())
    except Exception as exc:
        logger.exception("list report specs failed")
        raise HTTPException(status_code=503, detail=f"Failed to list report specs: {exc}") from exc


@router.get("/spec/{name}", response_model=ReportSpec)
def get_spec(name: str) -> ReportSpec:
    """API endpoint: GET /api/report/spec/{name} — load one saved report spec."""
    try:
        return ReportSpec.model_validate(load_spec(name))
    except ReportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("load report spec failed for name=%s", name)
        raise HTTPException(status_code=404, detail=f"Report spec '{name}' not found: {exc}") from exc


@router.put("/spec/{name}", response_model=ReportSpecSaveResult)
def put_spec(name: str, spec: ReportSpec) -> ReportSpecSaveResult:
    """API endpoint: PUT /api/report/spec/{name} — save a report spec to S3."""
    try:
        path = save_spec(name, spec.model_dump())
        saved_name = path.rsplit("/", 1)[-1][: -len(".json")]
        return ReportSpecSaveResult(name=saved_name, path=path, specs=list_specs())
    except ReportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("save report spec failed for name=%s", name)
        raise HTTPException(status_code=503, detail=f"Failed to save report spec: {exc}") from exc


@router.post("/references", response_model=ReferencesResponse)
def post_references(req: ReferencesRequest) -> ReferencesResponse:
    """API endpoint: POST /api/report/references — fetch reference pages.

    Resolves Confluence URLs to {url, title, text} (cached server-side) for the
    reference pills and as LLM context; non-Confluence URLs come back with empty
    text but are still listed.
    """
    try:
        from bituslabs_ds.confluence.references import load_references

        return ReferencesResponse(references=load_references(req.urls))
    except Exception as exc:
        logger.exception("load references failed")
        raise HTTPException(status_code=503, detail=f"Failed to load references: {exc}") from exc


@router.post("/description", response_model=GenerateProxyResponse)
def post_description(payload: Dict[str, Any] = Body(...)) -> GenerateProxyResponse:
    """API endpoint: POST /api/report/description — proxy to ai_agent.

    Forwards {data_summary, existing_description, references, language, model}
    to the agent service's LLM generation; in prod the ALB routes /api/report/*
    here, so this hop keeps the contract while the LLM stays on ai_agent.
    """
    try:
        return GenerateProxyResponse(**proxy_generate("description", payload))
    except ReportError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/summary", response_model=GenerateProxyResponse)
def post_summary(payload: Dict[str, Any] = Body(...)) -> GenerateProxyResponse:
    """API endpoint: POST /api/report/summary — proxy to ai_agent (see /description)."""
    try:
        return GenerateProxyResponse(**proxy_generate("summary", payload))
    except ReportError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/export", response_model=ExportResponse)
def post_export(req: ExportRequest) -> ExportResponse:
    """API endpoint: POST /api/report/export — write the report to Confluence.

    Receives client-rendered ECharts PNGs (base64) + descriptions + summary +
    references; attaches the images and idempotently replaces the
    "Dashboard Report" region of the target page (content above it preserved).
    """
    try:
        page_id, uploaded, version = export_report_pngs(
            confluence_url=req.confluence_url,
            summary=req.summary,
            figures=[f.model_dump() for f in req.figures],
            references=[r.model_dump() for r in req.references],
        )
        return ExportResponse(page_id=page_id, figures_uploaded=uploaded, page_version=version)
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("report export failed")
        raise HTTPException(status_code=503, detail=f"Export failed: {exc}") from exc
