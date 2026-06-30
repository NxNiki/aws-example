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
    assert "/api/report/specs" in paths
    assert "/api/report/spec/{name}" in paths
    assert "/api/report/references" in paths
    assert "/api/report/description" in paths
    assert "/api/report/summary" in paths
    assert "/api/report/export" in paths


def test_collect_window_keys_by_config_id_not_collide():
    """Regression: the window cache must isolate configs.

    The raw config YAML has no ``id`` of its own, so a cache key built from
    ``cfg.get("id")`` was ``None`` for every game — ss03's request hit ss02's
    cached user rows. ``load_raw_config`` now stamps the id; ``collect_window``
    refuses a missing one and keys distinctly per config.
    """
    from datetime import datetime

    import polars as pl
    import pytest

    from dashboard_api.services import common
    from dashboard_api.services.common import SeriesError, collect_window

    common._window_cache.clear()
    df_a = pl.DataFrame({"d": ["2026-06-01"], "v": [1]}).with_columns(pl.col("d").str.to_datetime())
    df_b = pl.DataFrame({"d": ["2026-06-01"], "v": [2]}).with_columns(pl.col("d").str.to_datetime())
    lo, hi = datetime(2026, 6, 1), datetime(2026, 6, 2)

    a = collect_window({"id": "ss02"}, "day", df_a.lazy(), "d", lo, hi)
    b = collect_window({"id": "ss03"}, "day", df_b.lazy(), "d", lo, hi)
    assert a["v"][0] == 1 and b["v"][0] == 2  # no cross-config bleed
    assert len(common._window_cache) == 2

    with pytest.raises(SeriesError):
        collect_window({}, "day", df_a.lazy(), "d", lo, hi)  # missing id must raise, not collide
    common._window_cache.clear()


def test_load_raw_config_stamps_id():
    from pathlib import Path

    from dashboard_api.services.configs import load_raw_config

    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "dashboard")
    cfg = load_raw_config(config_dir, "ss02")
    assert cfg is not None and cfg["id"] == "ss02"


def test_load_config_from_s3(monkeypatch, mock_s3_client):
    """config_dir may be an s3:// URI: configs are listed/parsed straight from S3 so a
    new dashboard_config-*.yaml needs only an upload, no image rebuild."""
    import yaml

    from bituslabs_ds import s3_utils
    from dashboard_api.services import configs

    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    # Rebind the cached s3 client (and config cache) to the moto-mocked backend.
    s3_utils._get_s3_client_for_pid.cache_clear()
    configs.clear_config_cache()

    mock_s3_client.put_object(
        Bucket="test-bucket",
        Key="dashboard-configs/dashboard_config-foo.yaml",
        Body=yaml.safe_dump({"title": "From S3", "stats_by_date": {"date_col": "d"}}),
    )
    mock_s3_client.put_object(Bucket="test-bucket", Key="dashboard-configs/notes.txt", Body=b"ignore me")

    config_dir = "s3://test-bucket/dashboard-configs"
    summaries = configs.list_config_summaries(config_dir)
    assert {s.id for s in summaries} == {"foo"}  # non-yaml ignored
    assert summaries[0].title == "From S3"

    cfg = configs.load_raw_config(config_dir, "foo")
    assert cfg is not None and cfg["id"] == "foo" and cfg["title"] == "From S3"
    assert configs.load_raw_config(config_dir, "missing") is None

    configs.clear_config_cache()
    s3_utils._get_s3_client_for_pid.cache_clear()
