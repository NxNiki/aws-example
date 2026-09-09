"""Report Spec persistence + the dashboard_api → ai_agent LLM proxy.

Specs are named JSON files in ``s3://{bucket}/report-specs/{name}.json`` (the
bucket has versioning, so every save is revertible — the agent-edit/undo story
from the redesign doc).

The description/summary endpoints PROXY to the ai_agent service: in production
the ALB routes ``/api/report/*`` to dashboard_api, and per the architecture the
LLM-bound generation is the single cross-service hop (dashboard_api →
``settings.agent_api_url``).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import list_s3_files, read_json_from_s3, write_json_to_s3
from dashboard_api.settings import settings

logger = logging.getLogger(__name__)

SPECS_PREFIX = "report-specs"
_AGENT_TIMEOUT_S = 120  # LLM generation can take a while (mirrors the legacy timeout)


class ReportError(Exception):
    """Invalid spec name / agent proxy failure."""


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", name or "").strip("._")
    if not cleaned:
        raise ReportError("Invalid report spec name")
    return cleaned


def _path(name: str) -> str:
    return f"s3://{S3_BUCKET}/{SPECS_PREFIX}/{_safe_name(name)}.json"


def list_specs() -> list[str]:
    uris = list_s3_files(S3_BUCKET, f"{SPECS_PREFIX}/", pattern=r"\.json$")
    return sorted(uri.rsplit("/", 1)[-1][: -len(".json")] for uri in uris)


def load_spec(name: str) -> dict[str, Any]:
    return read_json_from_s3(_path(name))


def save_spec(name: str, spec: dict[str, Any]) -> str:
    path = _path(name)
    write_json_to_s3(spec, path)
    return path


def proxy_generate(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Forward a description/summary request to ai_agent and return its JSON.

    ``endpoint`` is "description" or "summary". Raises ReportError with a
    user-facing message when the agent is unreachable or errors.
    """
    url = f"{settings.agent_api_url}/api/report/{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_AGENT_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:300]
        raise ReportError(f"Agent generation failed (HTTP {exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise ReportError(f"AI agent unreachable at {settings.agent_api_url}: {exc.reason}") from exc
