"""Smoke tests for the dashboard_api FastAPI app (Phase 0 foundation).

These exercise app construction and the OpenAPI contract the frontend client is
generated from, without S3/network or an HTTP client. They run with just the
``main,dashboard_api`` dependency set (the Docker runtime), so they make a fast,
reliable CI gate for the new service.
"""

from dashboard_api.app import app
from dashboard_api.routers.health import health


def test_health_ok():
    payload = health()
    assert payload["status"] == "ok"
    assert payload["service"] == "dashboard_api"


def test_openapi_exposes_data_routes():
    # The frontend's typed client is generated from this schema (make openapi),
    # so the data routes must stay present in the contract.
    paths = app.openapi()["paths"]
    assert "/api/health" in paths
    assert "/api/data/series" in paths
    assert "/api/data/configs" in paths
    assert "/api/data/config/{config_id}" in paths
    assert "/api/data/group-values" in paths
    assert "/api/data/date-bounds" in paths
    assert "/api/data/group-distribution" in paths
    assert "/api/data/deepdive" in paths
    assert "/api/data/deepdive-metrics" in paths
    assert "/api/views" in paths
    assert "/api/views/{name}" in paths
