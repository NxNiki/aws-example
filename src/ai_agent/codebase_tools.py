"""
Codebase search/read tools for the ai_agent ReAct agent.

Why this exists
---------------
Source code is a poor fit for the Confluence RAG index (different chunking
needs, fast churn, lots of low-signal hits on imports/boilerplate). Instead,
expose the *live* repo to the agent via grep + bounded file reads — same
mental model Claude Code itself uses. The agent decides when to look at code;
results always reflect the current checkout.

Safety
------
- All paths are resolved and verified to live under ``_REPO_ROOT``.
- Only directories in ``_ALLOWED_DIRS`` are searchable / readable. This
  excludes ``.env``, ``.git``, caches, deploy artifacts, and ``output_*``.
- Output is capped per-tool to keep agent context bounded.
- ``ripgrep`` is preferred when present (respects ``.gitignore`` and is
  ~100x faster on large trees); the Python fallback is intentionally simple.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool

from bituslabs_ds.config import LOCAL_ROOT

logger = logging.getLogger(__name__)

_REPO_ROOT = LOCAL_ROOT

# Directories the agent is allowed to search/read. Add to this list rather
# than widening to the repo root — keeps secrets, build artifacts, and
# large data dumps out of agent context.
_ALLOWED_DIRS = ("src", "jobs", "infra", "tests", "entry_points")

# Names skipped during the Python-fallback walk (ripgrep already honours .gitignore).
_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".ipynb_checkpoints",
}

# Extensions considered "source" for grep + list. Avoids matching parquet,
# pickles, and other binary blobs that occasionally sit under jobs/.
_SOURCE_EXTS = {
    ".py",
    ".pyi",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".sh",
    ".dockerfile",
    ".cfg",
    ".ini",
}

_MAX_GREP_HITS = 40
_MAX_FILE_BYTES = 64 * 1024  # 64 KB hard cap on a single read
_MAX_LIST_ENTRIES = 200


def _resolve_safe(rel_path: str) -> Optional[Path]:
    """Resolve ``rel_path`` against the repo root, returning ``None`` if it
    escapes the repo or lands outside ``_ALLOWED_DIRS``."""
    candidate = (_REPO_ROOT / rel_path).resolve()
    try:
        candidate.relative_to(_REPO_ROOT)
    except ValueError:
        return None
    rel = candidate.relative_to(_REPO_ROOT)
    if rel.parts and rel.parts[0] not in _ALLOWED_DIRS:
        return None
    return candidate


def _have_ripgrep() -> bool:
    return shutil.which("rg") is not None


@tool
def grep_codebase(pattern: str, path: str = "", file_glob: str = "") -> str:
    """Search the repository for a regex pattern. Returns matching lines with
    ``path:line: content``.

    Use this to locate where a symbol, SQL alias, config key, or string
    literal is defined or referenced — analogous to ``ripgrep`` on the repo.

    Search strategy — IMPORTANT:
    - Code uses snake_case / CamelCase tokens, not English phrases. Never
      grep for a multi-word natural-language phrase like ``"clip data"`` or
      ``"remove outliers"`` — it will return zero matches because no source
      file contains that literal string.
    - Start with the most distinctive SINGLE token from the user's question
      (``"clip"``, ``"outlier"``, ``"weighted_average"``, ``"BOOST_POOL"``).
      If that's too noisy, narrow with ``\\b<token>\\b`` for word-boundary
      match, or pass ``path=`` to scope to one module (``"src/dashboards"``),
      or ``file_glob=`` to filter file types (``"*.py"``).
    - If the first single-token search returns nothing, try variants before
      giving up: stem (``"clipping"`` → ``"clip"``), the verb form
      (``"clip"``), the function-style form (``"clip_outliers"``), or the
      Python module name from the dashboard config.

    Args:
        pattern: Regex (extended). Examples: ``"def build_metadata"``,
                 ``"ACTIVE_USER_MIN_BETS\\s*="``, ``"user_total_bet"``.
                 NOT ``"clip data"`` — that is a user phrase, not a code
                 token.
        path: Optional sub-path to scope the search (e.g. ``"src/ai_agent"``,
              ``"jobs/fish_hunter"``). Must live under one of
              ``src/``, ``jobs/``, ``infra/``, ``tests/``, ``entry_points/``.
        file_glob: Optional ripgrep ``--glob`` filter (e.g. ``"*.py"``,
                   ``"*.sql"``). Ignored by the Python fallback.
    """
    search_root = _resolve_safe(path) if path else _REPO_ROOT
    if search_root is None:
        return f"Path '{path}' is outside the searchable area ({', '.join(_ALLOWED_DIRS)})."

    if _have_ripgrep():
        cmd = ["rg", "--no-heading", "--line-number", "--max-count", "5", "--color", "never"]
        if file_glob:
            cmd += ["--glob", file_glob]
        # If no path was given, restrict ripgrep to the allowed dirs that
        # actually exist in this checkout — the runtime image is a subset of
        # the repo (no infra/, tests/, entry_points/), so passing the full
        # allowlist would emit spurious "No such file or directory" warnings.
        if path:
            targets = [str(search_root)]
        else:
            targets = [str(_REPO_ROOT / d) for d in _ALLOWED_DIRS if (_REPO_ROOT / d).exists()]
        cmd += [pattern, *targets]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except (subprocess.TimeoutExpired, OSError) as exc:
            return f"grep failed: {exc}"
        if out.returncode not in (0, 1):  # 1 = no matches, still fine
            return f"grep error: {out.stderr.strip() or 'unknown'}"
        lines = out.stdout.splitlines()
    else:
        lines = _python_grep(pattern, search_root if path else None)

    if not lines:
        return f"No matches for /{pattern}/."

    # Make paths repo-relative for readability and cap output.
    formatted: List[str] = []
    for raw in lines[:_MAX_GREP_HITS]:
        try:
            file_part, rest = raw.split(":", 1)
            rel = Path(file_part).resolve().relative_to(_REPO_ROOT)
            formatted.append(f"{rel}:{rest}")
        except (ValueError, OSError):
            formatted.append(raw)

    suffix = ""
    if len(lines) > _MAX_GREP_HITS:
        suffix = f"\n... and {len(lines) - _MAX_GREP_HITS} more matches (narrow the search)."
    return "\n".join(formatted) + suffix


def _python_grep(pattern: str, root: Optional[Path]) -> List[str]:
    """Fallback when ripgrep isn't installed. Plain substring match — keeps
    the fallback path dependency-free; ripgrep handles real regex."""
    import re

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return [f"<regex error: {exc}>"]

    roots = [root] if root else [_REPO_ROOT / d for d in _ALLOWED_DIRS]
    hits: List[str] = []
    for r in roots:
        if not r.exists():
            continue
        for fp in r.rglob("*"):
            if len(hits) >= _MAX_GREP_HITS * 2:
                return hits
            if not fp.is_file() or fp.suffix not in _SOURCE_EXTS:
                continue
            if any(part in _SKIP_DIR_NAMES for part in fp.parts):
                continue
            try:
                with fp.open("r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if regex.search(line):
                            hits.append(f"{fp}:{lineno}:{line.rstrip()}")
            except OSError:
                continue
    return hits


@tool
def read_source_file(rel_path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read a slice of a source file from the repo.

    Use this after ``grep_codebase`` returns a hit you want to inspect, or
    when ``lookup_column`` / ``read_etl_source`` points at code you need to
    understand more broadly.

    Args:
        rel_path: Path relative to the repo root (e.g.
                  ``"src/ai_agent/chat_agent.py"``). Must live under one of
                  ``src/``, ``jobs/``, ``infra/``, ``tests/``, ``entry_points/``.
        start_line: 1-indexed first line to include (default 1).
        end_line: 1-indexed last line to include. 0 means "until EOF or the
                  64KB cap, whichever comes first".
    """
    target = _resolve_safe(rel_path)
    if target is None:
        return f"Path '{rel_path}' is outside the readable area ({', '.join(_ALLOWED_DIRS)})."
    if not target.is_file():
        return f"'{rel_path}' is not a file."
    if target.stat().st_size > _MAX_FILE_BYTES and end_line == 0:
        return (
            f"'{rel_path}' is {target.stat().st_size} bytes (> {_MAX_FILE_BYTES}). "
            "Pass start_line/end_line to read a specific slice."
        )

    try:
        with target.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return f"Failed to read '{rel_path}': {exc}"

    if start_line < 1:
        start_line = 1
    end = len(lines) if end_line <= 0 else min(end_line, len(lines))
    selected = lines[start_line - 1 : end]
    rendered = "".join(f"{start_line + i:>5}: {ln}" for i, ln in enumerate(selected))

    # Hard byte cap on the rendered output too — protects against pathological lines.
    if len(rendered) > _MAX_FILE_BYTES:
        rendered = rendered[:_MAX_FILE_BYTES] + "\n... (truncated)"
    header = f"{rel_path} (lines {start_line}-{end} of {len(lines)})\n"
    return header + rendered


@tool
def list_source_files(path: str = "", file_glob: str = "*.py") -> str:
    """List source files under a path in the repo.

    Use this to discover what lives in a module before grepping or reading.

    Args:
        path: Sub-path under the repo (e.g. ``"src/ai_agent"``,
              ``"jobs/cluster_analysis"``). Defaults to listing the top of
              each allowed directory.
        file_glob: Filename glob (default ``"*.py"``). Examples:
                   ``"*.sql"``, ``"*.yaml"``, ``"*"``.
    """
    target = _resolve_safe(path) if path else _REPO_ROOT
    if target is None:
        return f"Path '{path}' is outside the listable area ({', '.join(_ALLOWED_DIRS)})."

    roots = [target] if path else [_REPO_ROOT / d for d in _ALLOWED_DIRS if (_REPO_ROOT / d).exists()]
    matches: List[str] = []
    for r in roots:
        if not r.exists():
            continue
        for fp in sorted(r.rglob(file_glob)):
            if not fp.is_file():
                continue
            if any(part in _SKIP_DIR_NAMES for part in fp.parts):
                continue
            matches.append(str(fp.relative_to(_REPO_ROOT)))
            if len(matches) >= _MAX_LIST_ENTRIES:
                break
        if len(matches) >= _MAX_LIST_ENTRIES:
            break

    if not matches:
        return f"No files matching '{file_glob}' under '{path or 'allowed dirs'}'."
    suffix = ""
    if len(matches) >= _MAX_LIST_ENTRIES:
        suffix = f"\n... (capped at {_MAX_LIST_ENTRIES}; narrow path or file_glob)."
    return "\n".join(matches) + suffix


CODEBASE_TOOLS = [grep_codebase, read_source_file, list_source_files]
