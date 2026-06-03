"""Dump the dashboard_api OpenAPI schema as JSON.

Single source of truth for the frontend's typed client: this schema is fed to
`openapi-typescript` to generate `frontend/src/api/schema.d.ts`. Regenerate
both with `make openapi` (or `scripts/gen_openapi_client.sh`).

Usage:
    python -m dashboard_api.openapi              # write JSON to stdout
    python -m dashboard_api.openapi out.json     # write JSON to a file
"""

from __future__ import annotations

import json
import sys
from typing import Optional, Sequence

from dashboard_api.app import create_app


def dump(path: Optional[str] = None) -> str:
    schema = create_app().openapi()
    # sort_keys for deterministic output → clean diffs when the schema changes.
    text = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    out = args[0] if args else None
    text = dump(out)
    if out:
        sys.stderr.write(f"wrote OpenAPI schema to {out}\n")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
