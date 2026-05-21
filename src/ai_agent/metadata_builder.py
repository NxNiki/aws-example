"""
Build and cache a unified metadata document for the AI chat agent.

Combines:
  1. Hand-curated column_metadata.yaml (business meaning, group definitions)
  2. ETL source code scanning (SQL computation logic, column aliases)
  3. Dashboard config YAML (which columns are displayed, game context)

The merged result is persisted as a JSON cache file so subsequent questions
can be answered without re-scanning ETL files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent  # src/ai_agent/
_SRC_DIR = _THIS_DIR.parent  # src/
_REPO_ROOT = _SRC_DIR.parent  # project root
_DASHBOARDS_DIR = _SRC_DIR / "dashboards"  # for cross-package file reads

METADATA_YAML_PATH = _THIS_DIR / "column_metadata.yaml"
CACHE_DIR = _THIS_DIR / ".metadata_cache"
CACHE_FILE = CACHE_DIR / "metadata_cache.json"


def _file_hash(path: Path) -> str:
    """SHA-256 hex digest of file contents (for cache invalidation)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_sql_snippets(source_code: str) -> List[str]:
    """
    Pull SQL fragments from triple-quoted strings and dedent() calls.
    Returns a list of SQL snippets found in the file.
    """
    snippets: List[str] = []
    for m in re.finditer(
        r'(?:dedent\s*\(\s*)?(?:f?"""(.*?)"""|f?\'\'\'(.*?)\'\'\')',
        source_code,
        re.DOTALL,
    ):
        text = m.group(1) or m.group(2) or ""
        sql_keywords = {"SELECT", "FROM", "WHERE", "GROUP BY", "JOIN", "WITH", "CASE"}
        upper = text.upper()
        if sum(1 for kw in sql_keywords if kw in upper) >= 2:
            snippets.append(text.strip())
    return snippets


def _extract_column_aliases(sql: str) -> Dict[str, str]:
    """
    Extract ``expression AS alias`` mappings from SQL text.
    Returns {alias: expression_snippet}.
    """
    aliases: Dict[str, str] = {}
    for m in re.finditer(
        r"(.{5,120}?)\s+AS\s+(\w+)",
        sql,
        re.IGNORECASE,
    ):
        expr = m.group(1).strip().lstrip(",").strip()
        alias = m.group(2).strip()
        if alias.upper() not in ("DATE", "VARCHAR", "INT", "FLOAT", "BOOLEAN"):
            aliases[alias] = expr
    return aliases


def _scan_etl_file(path: Path) -> Dict[str, Any]:
    """Scan a single ETL Python file and extract metadata."""
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not read ETL file %s: %s", path, exc)
        return {}

    sql_snippets = _extract_sql_snippets(source)
    all_aliases: Dict[str, str] = {}
    for snippet in sql_snippets:
        all_aliases.update(_extract_column_aliases(snippet))

    game_match = re.search(r'GAME_ID\s*=\s*["\'](\w+)["\']', source)
    game_id = game_match.group(1) if game_match else None

    return {
        "file": str(path.relative_to(_REPO_ROOT)),
        "game_id": game_id,
        "sql_column_aliases": all_aliases,
        "sql_snippets_count": len(sql_snippets),
        "sql_snippets": sql_snippets,
    }


def _scan_python_aggregation_file(path: Path) -> Dict[str, str]:
    """Scan a Python file for aggregation logic (cached_property methods with column names).

    Returns {column_name: code_snippet} for each ``@cached_property`` method
    whose name starts with ``_`` and produces a dashboard column.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}

    results: Dict[str, str] = {}
    # Match @cached_property blocks: method name + body up to next @cached_property or class-level def
    pattern = re.compile(
        r"@cached_property\s*\n\s+def\s+(_\w+)\(self\).*?(?=\n\s+@|\n\s+def\s+[^_]|\nclass\s|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(source):
        method_name = m.group(1)
        col_name = method_name.lstrip("_")
        body = m.group(0).strip()
        # Limit to reasonable size
        if len(body) > 2000:
            body = body[:2000] + "\n    # ... (truncated)"
        results[col_name] = body

    # Also extract class-level constants that affect aggregation
    constants: List[str] = []
    for cm in re.finditer(r"^\s+(\w+):\s+ClassVar\[.+?\]\s*=\s*(.+)", source, re.MULTILINE):
        constants.append(f"{cm.group(1)} = {cm.group(2).strip()}")

    if constants:
        results["__class_constants__"] = "\n".join(constants)

    return results


def load_metadata_yaml() -> Dict[str, Any]:
    """Load and return the hand-curated column_metadata.yaml."""
    with open(METADATA_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dashboard_config(config_path: Path) -> Dict[str, Any]:
    """Load a dashboard config YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_metadata(
    *,
    dashboard_config_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Build or load the unified metadata document.

    If a valid cache exists and source files haven't changed, returns the
    cached version. Otherwise rescans ETL files and rebuilds.

    Parameters
    ----------
    dashboard_config_path : Path, optional
        Path to the active dashboard config YAML. If provided, its columns
        and game context are merged into the result.
    force_rebuild : bool
        Skip cache and rebuild from scratch.

    Returns
    -------
    dict
        Unified metadata document with keys:
        - ``columns``: merged column definitions (YAML + ETL-derived)
        - ``groups``: user group definitions
        - ``games``: game context
        - ``etl_sources``: per-ETL-file scan results
        - ``dashboard_config``: active config info (if provided)
    """
    meta_yaml = load_metadata_yaml()

    source_hashes: Dict[str, str] = {
        "column_metadata.yaml": _file_hash(METADATA_YAML_PATH),
    }
    etl_files: List[Path] = []
    for rel in meta_yaml.get("etl_files", []):
        p = _REPO_ROOT / rel
        if p.exists():
            etl_files.append(p)
            source_hashes[rel] = _file_hash(p)

    if dashboard_config_path and dashboard_config_path.exists():
        source_hashes["dashboard_config"] = _file_hash(dashboard_config_path)

    if not force_rebuild and CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if cached.get("source_hashes") == source_hashes:
                logger.info("Metadata cache is valid, using cached version")
                return cached["metadata"]
        except Exception:
            pass

    logger.info("Building metadata from %d ETL files", len(etl_files))

    etl_results = [_scan_etl_file(p) for p in etl_files]

    etl_aliases: Dict[str, Dict[str, str]] = {}
    for result in etl_results:
        for alias, expr in result.get("sql_column_aliases", {}).items():
            if alias not in etl_aliases:
                etl_aliases[alias] = {}
            src = result.get("file", "unknown")
            etl_aliases[alias][src] = expr

    # Scan Python aggregation logic (user_stats_aggregates.py lives in
    # the dashboards package — it's dashboard runtime code that the chat
    # agent only introspects for documenting aggregation methods).
    agg_file = _DASHBOARDS_DIR / "user_stats_aggregates.py"
    python_agg: Dict[str, str] = {}
    if agg_file.exists():
        python_agg = _scan_python_aggregation_file(agg_file)
        logger.info("Scanned %d aggregation methods from %s", len(python_agg), agg_file.name)

    columns = dict(meta_yaml.get("columns", {}))
    for alias, sources in etl_aliases.items():
        if alias not in columns:
            columns[alias] = {
                "category": "etl_derived",
                "description": "Computed in ETL. See source files for full SQL.",
                "etl_formula": sources,
            }
        else:
            columns[alias].setdefault("etl_formula", {})
            if isinstance(columns[alias].get("etl_formula"), str):
                columns[alias]["etl_formula"] = {"note": columns[alias]["etl_formula"]}
            columns[alias]["etl_formula"].update(sources)

    # Attach Python aggregation logic to matching columns
    for col_name, code_snippet in python_agg.items():
        if col_name == "__class_constants__":
            continue
        if col_name in columns:
            columns[col_name]["python_aggregation"] = code_snippet
        else:
            columns[col_name] = {
                "category": "aggregate_metric",
                "description": "Computed in Python aggregation. See code for full logic.",
                "python_aggregation": code_snippet,
            }

    dashboard_info: Optional[Dict[str, Any]] = None
    if dashboard_config_path and dashboard_config_path.exists():
        dashboard_info = load_dashboard_config(dashboard_config_path)

    metadata: Dict[str, Any] = {
        "columns": columns,
        "groups": meta_yaml.get("groups", {}),
        "games": meta_yaml.get("games", {}),
        "etl_sources": etl_results,
        "python_aggregation_constants": python_agg.get("__class_constants__", ""),
        "dashboard_config": dashboard_info,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_doc = {"source_hashes": source_hashes, "metadata": metadata}
    CACHE_FILE.write_text(json.dumps(cache_doc, indent=2, default=str), encoding="utf-8")
    logger.info("Metadata cache written to %s", CACHE_FILE)

    return metadata
