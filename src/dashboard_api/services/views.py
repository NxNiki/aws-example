"""Save / load named dashboard *view snapshots* to S3.

The React-era replacement for the legacy Dash "save/load config" feature: it
persists a snapshot of the dashboard's UI selections (which game config,
granularity, date windows, cohorts, per-panel metrics / modes / clip) so a user
can recover a setup. The snapshot is OPAQUE to the backend — its shape is owned
by the frontend store; here we only list / read / write the JSON. (Report
content is a separate concern — the future Report Spec primitive — not this.)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import list_s3_files, read_json_from_s3, write_json_to_s3

logger = logging.getLogger(__name__)

VIEWS_PREFIX = "dashboard-views"


class ViewError(Exception):
    """Raised for an invalid view name."""


def _safe_name(name: str) -> str:
    """Sanitize a user-supplied view name into a safe S3 object stem (no path
    traversal, filesystem-safe)."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", name or "").strip("._")
    if not cleaned:
        raise ViewError("Invalid view name")
    return cleaned


def _path(name: str) -> str:
    return f"s3://{S3_BUCKET}/{VIEWS_PREFIX}/{_safe_name(name)}.json"


def list_views() -> list[str]:
    """Names of saved view snapshots (sorted)."""
    uris = list_s3_files(S3_BUCKET, f"{VIEWS_PREFIX}/", pattern=r"\.json$")
    names = [uri.rsplit("/", 1)[-1][: -len(".json")] for uri in uris]
    return sorted(names)


def load_view(name: str) -> dict[str, Any]:
    """Read one saved view snapshot."""
    return read_json_from_s3(_path(name))


def save_view(name: str, snapshot: dict[str, Any]) -> str:
    """Persist a view snapshot under the (sanitized) name, overwriting if present.
    Returns the s3:// path it was written to."""
    path = _path(name)
    write_json_to_s3(snapshot, path)
    return path
