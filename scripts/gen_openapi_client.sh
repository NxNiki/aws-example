#!/usr/bin/env bash
# Regenerate the dashboard_api OpenAPI schema and the frontend's typed client.
#
#   bash scripts/gen_openapi_client.sh   (or: make openapi)
#
# Run after changing any dashboard_api request/response model so the schema
# (frontend/openapi.json) and the generated types (frontend/src/api/schema.d.ts)
# stay in sync with the Pydantic models. Run from the repo root.
set -euo pipefail

if [[ ! -f "pyproject.toml" ]]; then
  echo "error: run from the project root (pyproject.toml not found)" >&2
  exit 1
fi

PYTHONPATH=src python -m dashboard_api.openapi frontend/openapi.json
npm --prefix frontend run gen:types
echo "OpenAPI schema + TS client regenerated."
