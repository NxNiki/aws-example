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


def test_percentile_filter_drops_tails():
    """The Stats-by-Group / Deep Dive "filter [min]% [max]%" control removes
    samples outside the percentile bounds (unlike clip, which pins them)."""
    import numpy as np

    from dashboard_api.services.deepdive import _percentile_filter, _percentile_row_mask
    from dashboard_api.services.group_distribution import _percentile_filter as _gd_filter

    vals = np.arange(101, dtype=float)  # 0..100: value == its own percentile
    for fn in (_percentile_filter, _gd_filter):
        out = fn(vals, True, 10, 90)
        assert out.min() == 10 and out.max() == 90 and len(out) == 81
        assert len(fn(vals, False, 10, 90)) == 101  # disabled → untouched
        assert len(fn(vals, True, None, None)) == 101  # no bounds → untouched
        assert fn(vals, True, 0, 100).tolist() == vals.tolist()  # full range keeps all
        assert len(fn(np.array([]), True, 10, 90)) == 0  # empty stays empty

    one_sided = _percentile_filter(vals, True, None, 50)
    assert one_sided.max() == 50 and one_sided.min() == 0

    # Row mask: a row is kept only when EVERY column is inside its own bounds.
    arr = np.column_stack([vals, vals[::-1]])
    mask = _percentile_row_mask(arr, 10, 90)
    kept = arr[mask]
    assert kept[:, 0].min() >= 10 and kept[:, 0].max() <= 90
    assert kept[:, 1].min() >= 10 and kept[:, 1].max() <= 90


def _lifecycle_fixture():
    """4 user-day rows: u1 at day 0 and 2, u2 at day 40, u3 never bet (no first-bet)."""
    import polars as pl

    df = pl.DataFrame(
        {
            "d": ["2026-06-01", "2026-06-03", "2026-06-10", "2026-06-10"],
            "user_id": ["u1", "u1", "u2", "u3"],
            "ai_group": ["A", "A", "A", "A"],
            "life_cycle_group": ["new", "new", "old", "old"],
        }
    ).with_columns(pl.col("d").str.to_datetime())
    first = pl.DataFrame({"user_id": ["u1", "u2"], "first_bet_date": ["2026-06-01", "2026-05-01"]}).with_columns(
        pl.col("first_bet_date").str.to_datetime()
    )
    return df, first


def test_iter_cohorts_lifecycle_groups(monkeypatch):
    """Custom lifecycle groups redefine life_cycle_group by the HALF-OPEN period range
    [start, end) since first bet: labels come from the request, ranges may
    overlap, end=None is open-ended, 'all' is no-filter, and users without a
    first bet fall only into 'all'."""
    from dashboard_api.services import common

    df, first = _lifecycle_fixture()
    monkeypatch.setattr(common, "load_first_bet_dates", lambda cfg: first)
    cfg = {"id": "g", "stats_by_date": {"user_group_cols": ["ai_group", "life_cycle_group"]}}
    lifecycle = [
        {"label": "day0", "start": 0, "end": 1},
        {"label": "day0-3", "start": 0, "end": 4},  # overlaps day0 on purpose
        {"label": "rest", "start": 4, "end": None},
        {"label": "all", "start": 0, "end": None},
    ]
    out = dict(common.iter_cohorts(cfg, df, {"ai_group": ["A"]}, lifecycle=lifecycle, date_col="d"))

    # cohort_label collapses "all" (same as the stored-label pickers): "A | all" → "A".
    assert set(out) == {"A | day0", "A | day0-3", "A | rest", "A"}
    assert set(out["A | day0"]["user_id"]) == {"u1"} and out["A | day0"].height == 1
    assert out["A | day0-3"].height == 2  # both u1 rows: overlapping ranges both keep them
    assert set(out["A | rest"]["user_id"]) == {"u2"}
    assert out["A"].height == 4  # "all" = no filter — includes u3, who never bet
    # The derived period column rides along for DataMetrics (num_new_users).
    assert all(common.PERIODS_COL in c.columns for c in out.values())


def test_iter_cohorts_lifecycle_week_month_use_period_index(monkeypatch):
    """On weekly/monthly granularity the range unit is the calendar week/month
    index since the user's first bet (a week/month row aggregates the whole
    period, so day units would slice by start weekday). Week 0 = the week of
    the first bet even when the row's week-start precedes a mid-week first
    bet; users with no first bet (null) still match only 'all'. Ranges are
    half-open: [0,1) = week 0 only, [1,2) = week 1 only."""
    import polars as pl

    from dashboard_api.services import common

    # Week rows (period start Monday 06-01 / 06-08); u1 first bet Thu 06-04.
    df = pl.DataFrame(
        {
            "d": ["2026-06-01", "2026-06-08", "2026-06-01"],
            "user_id": ["u1", "u1", "u9"],  # u9 never bet
            "life_cycle_group": ["new", "beginner", "old"],
        }
    ).with_columns(pl.col("d").str.to_datetime())
    first = pl.DataFrame({"user_id": ["u1"], "first_bet_date": ["2026-06-04"]}).with_columns(
        pl.col("first_bet_date").str.to_datetime()
    )
    monkeypatch.setattr(common, "load_first_bet_dates", lambda cfg: first)
    cfg = {"id": "g", "stats_by_date": {"user_group_cols": ["life_cycle_group"]}}
    lifecycle = [
        {"label": "new", "start": 0, "end": 1},
        {"label": "beginner", "start": 1, "end": 2},
        {"label": "all", "start": 0, "end": None},
    ]
    out = dict(common.iter_cohorts(cfg, df, {}, lifecycle=lifecycle, date_col="d", granularity="week"))
    assert out["new"]["d"].dt.strftime("%Y-%m-%d").to_list() == ["2026-06-01"]  # week 0 (first-bet week)
    assert out["beginner"]["d"].dt.strftime("%Y-%m-%d").to_list() == ["2026-06-08"]  # week 1
    assert out["all"].height == 3  # u9 (null periods) only appears here

    # Month rows; u1 first bet 06-20 → 06-01 row is month 0, 07-01 row month 1.
    dfm = pl.DataFrame({"d": ["2026-06-01", "2026-07-01"], "user_id": ["u1", "u1"], "life_cycle_group": ["new", "old"]})
    dfm = dfm.with_columns(pl.col("d").str.to_datetime())
    firstm = pl.DataFrame({"user_id": ["u1"], "first_bet_date": ["2026-06-20"]}).with_columns(
        pl.col("first_bet_date").str.to_datetime()
    )
    monkeypatch.setattr(common, "load_first_bet_dates", lambda cfg: firstm)
    outm = dict(common.iter_cohorts(cfg, dfm, {}, lifecycle=lifecycle, date_col="d", granularity="month"))
    assert outm["new"]["d"].dt.strftime("%Y-%m-%d").to_list() == ["2026-06-01"]
    assert outm["beginner"]["d"].dt.strftime("%Y-%m-%d").to_list() == ["2026-07-01"]


def test_iter_cohorts_without_lifecycle_uses_stored_labels(monkeypatch):
    """No lifecycle_groups in the request → unchanged behavior: equality filter
    on the ETL-stored life_cycle_group labels."""
    from dashboard_api.services import common

    df, _ = _lifecycle_fixture()
    monkeypatch.setattr(
        common, "load_first_bet_dates", lambda cfg: (_ for _ in ()).throw(AssertionError("must not derive"))
    )
    cfg = {"id": "g", "stats_by_date": {"user_group_cols": ["ai_group", "life_cycle_group"]}}
    out = dict(common.iter_cohorts(cfg, df, {"life_cycle_group": ["new"]}))
    assert set(out) == {"new"}
    assert set(out["new"]["user_id"]) == {"u1"}

    # A config without a life_cycle_group dimension ignores lifecycle entirely.
    cfg2 = {"id": "g2", "stats_by_date": {"user_group_cols": ["ab_group"]}}
    out2 = dict(common.iter_cohorts(cfg2, df, {}, lifecycle=[{"label": "day0", "start": 0, "end": 1}]))
    assert set(out2) == {"all"} and out2["all"].height == 4


def test_iter_cohorts_all_respects_group_col_partition(monkeypatch):
    """The ss-game ETLs UNION ALL every bet into a combined AB-test label AND a
    per-mathtable re-partition of the same bets, so 'all' on ai_group must
    aggregate only the configured disjoint labels — summing every row counted
    bets twice (the total_bet inflation bug). Named groups stay untouched."""
    import polars as pl

    from dashboard_api.services import common

    # u1's bets appear twice: once combined ('AI'/'Default'), once per-mathtable.
    df = pl.DataFrame(
        {
            "d": ["2026-06-01"] * 4,
            "user_id": ["u1", "u2", "u1", "u2"],
            "ai_group": ["AI", "Default", "mt_a", "Default_mt_a"],
            "user_total_bet": [100.0, 50.0, 100.0, 50.0],
        }
    ).with_columns(pl.col("d").str.to_datetime())
    cfg = {
        "id": "g",
        "stats_by_date": {
            "group_col": "ai_group",
            "group_col_partition": ["AI", "Default"],
            "user_group_cols": ["ai_group", "life_cycle_group"],
        },
    }
    out = dict(common.iter_cohorts(cfg, df, {}))
    assert out["all"]["user_total_bet"].sum() == 150.0  # not 300 — no double count
    assert set(out["all"]["ai_group"]) == {"AI", "Default"}

    # Named groups (combined or re-partition) are unchanged.
    named = dict(common.iter_cohorts(cfg, df, {"ai_group": ["AI", "mt_a"]}))
    assert named["AI"]["user_total_bet"].sum() == 100.0
    assert named["mt_a"]["user_total_bet"].sum() == 100.0

    # Composes with lifecycle groups on the second cohort column.
    first = pl.DataFrame({"user_id": ["u1", "u2"], "first_bet_date": ["2026-06-01", "2026-05-01"]}).with_columns(
        pl.col("first_bet_date").str.to_datetime()
    )
    monkeypatch.setattr(common, "load_first_bet_dates", lambda cfg: first)
    lc = dict(
        common.iter_cohorts(
            cfg, df, {}, lifecycle=[{"label": "new", "start": 0, "end": 3}], date_col="d", granularity="day"
        )
    )
    assert lc["new"]["user_total_bet"].sum() == 100.0  # u1 only, combined row only

    # Configs without a partition keep the old no-filter 'all'.
    cfg2 = {"id": "g2", "stats_by_date": {"group_col": "ai_group", "user_group_cols": ["ai_group"]}}
    assert dict(common.iter_cohorts(cfg2, df, {}))["all"].height == 4


def test_iter_cohorts_grain_collapses_user_rows(monkeypatch):
    """Two-column grain (ab_group × mathtable): each bet lives in exactly one
    row, so group sums are always right, but per-user stats must collapse the
    grain to one row per user whenever a grain dimension is unselected —
    summing additive components and recomputing ratios from the sums."""
    import polars as pl

    from dashboard_api.services import common

    # u1 played two mathtables on 06-01 (2 grain rows); u2 one mathtable.
    df = pl.DataFrame(
        {
            "d": ["2026-06-01"] * 3,
            "user_id": ["u1", "u1", "u2"],
            "ab_group": ["AI", "AI", "Default"],
            "mathtable": ["mt_a", "mt_b", "mt_a"],
            "user_num_bets": [10, 30, 5],
            "user_total_bet": [100.0, 300.0, 50.0],
            "user_total_payout": [90.0, 330.0, 40.0],
            "user_rtp": [0.9, 1.1, 0.8],
            "user_avg_bet_amount": [10.0, 10.0, 10.0],
        }
    ).with_columns(pl.col("d").str.to_datetime())
    cfg = {
        "id": "g",
        "stats_by_date": {
            "user_group_cols": ["ab_group", "mathtable", "life_cycle_group"],
            "user_row_grain": ["ab_group", "mathtable"],
        },
    }
    first = pl.DataFrame({"user_id": ["u1", "u2"], "first_bet_date": ["2026-06-01", "2026-05-01"]}).with_columns(
        pl.col("first_bet_date").str.to_datetime()
    )
    monkeypatch.setattr(common, "load_first_bet_dates", lambda cfg: first)

    out = dict(common.iter_cohorts(cfg, df, {}, date_col="d"))
    assert set(out) == {"all"}
    allc = out["all"].sort("user_id")
    assert allc.height == 2  # one row per user, not per (user, mathtable)
    u1 = allc.filter(pl.col("user_id") == "u1")
    assert u1["user_num_bets"][0] == 40 and u1["user_total_bet"][0] == 400.0
    assert abs(u1["user_rtp"][0] - 420.0 / 400.0) < 1e-9  # recomputed from sums
    assert abs(u1["user_avg_bet_amount"][0] - 10.0) < 1e-9

    # SQL semantics: an all-null sum stays null (a user with no BASE bets keeps
    # user_total_bet_bg null — excluded from per-user means — not 0).
    df_null = pl.DataFrame(
        {
            "d": ["2026-06-01", "2026-06-01"],
            "user_id": ["u1", "u1"],
            "ab_group": ["AI", "AI"],
            "mathtable": ["mt_a", "mt_b"],
            "user_num_bets": [10, 30],
            "user_total_bet": [100.0, 300.0],
            "user_total_bet_bg": [None, None],
        }
    ).with_columns(pl.col("d").str.to_datetime())
    collapsed = common.collapse_user_rows(df_null.drop(["ab_group", "mathtable"]), "d")
    assert collapsed["user_total_bet_bg"][0] is None
    assert collapsed["user_total_bet"][0] == 400.0

    # Pinning every grain column keeps the stored rows untouched.
    pinned = dict(common.iter_cohorts(cfg, df, {"ab_group": ["AI"], "mathtable": ["mt_a"]}, date_col="d"))
    assert set(pinned) == {"AI | mt_a"}
    assert pinned["AI | mt_a"].height == 1 and pinned["AI | mt_a"]["user_rtp"][0] == 0.9

    # Three active dimensions cross with lifecycle groups.
    lc = dict(
        common.iter_cohorts(
            cfg,
            df,
            {"ab_group": ["AI"]},
            lifecycle=[{"label": "new", "start": 0, "end": 3}],
            date_col="d",
            granularity="day",
        )
    )
    assert set(lc) == {"AI | new"}
    assert lc["AI | new"].height == 1 and lc["AI | new"]["user_num_bets"][0] == 40  # u1 collapsed


def test_iter_cohorts_range_groups_bucket_by_value(monkeypatch):
    """The range-group dimension (e.g. fish_value) buckets rows by INCLUSIVE
    [min, max] ranges from the request; a pinned range still collapses the
    grain (one range spans many stored values), and 'all' is no filter."""
    import polars as pl

    from dashboard_api.services import common

    df = pl.DataFrame(
        {
            "d": ["2026-06-01"] * 3,
            "user_id": ["u1", "u1", "u2"],
            "ab_test_group": ["DEFAULT_FALLBACK"] * 3,
            "fish_value": [10, 130, 500],
            "user_num_bets": [5, 20, 7],
            "user_total_bet": [50.0, 200.0, 70.0],
            "user_max_profit": [10.0, 90.0, 30.0],
        }
    ).with_columns(pl.col("d").str.to_datetime())
    cfg = {
        "id": "fh",
        "stats_by_date": {
            "user_group_cols": ["ab_test_group", "fish_value"],
            "user_row_grain": ["fish_value"],
            "range_group": {"column": "fish_value", "name": "Fish level"},
        },
    }
    groups = [
        {"label": "low", "min": 0, "max": 10},
        {"label": "low-med", "min": 0, "max": 130},  # overlap allowed
        {"label": "ultra", "min": 201, "max": None},
        {"label": "all", "min": 0, "max": None},
    ]
    out = dict(common.iter_cohorts(cfg, df, {}, date_col="d", range_groups=groups))
    assert set(out) == {"low", "low-med", "ultra", "all"}
    assert out["low"]["user_total_bet"].to_list() == [50.0]  # inclusive max: fish_value 10 kept
    lowmed = out["low-med"].sort("user_id")
    assert lowmed.height == 1 and lowmed["user_num_bets"][0] == 25  # u1's two rows collapsed
    assert lowmed["user_max_profit"][0] == 90.0  # MAX combinator
    assert out["ultra"]["user_id"].to_list() == ["u2"]
    assert out["all"].sort("user_id")["user_num_bets"].to_list() == [25, 7]


def test_series_request_accepts_ranges():
    """The global Date-groups picker sends up to three windows to /series; old
    clients (and saved report recipes) still use date_from/date_to."""
    from dashboard_api.schemas.data import Series, SeriesRequest

    req = SeriesRequest(config="c", metrics=["m"])
    assert req.ranges is None  # absent → single-window behavior
    req2 = SeriesRequest(
        config="c",
        metrics=["m"],
        ranges=[{"start": "2026-01-01", "end": "2026-01-31"}, {"start": "2026-06-01", "end": "2026-06-30"}],
    )
    assert req2.ranges is not None and len(req2.ranges) == 2
    s = Series(metric="m", cohort="all", kind="raw", additive=False, x=[], y=[], lower=[], upper=[])
    assert s.range_index is None and s.range_label is None  # optional for old responses


def test_load_series_ranges_tag_series(monkeypatch):
    """With ranges, load_series emits one tagged series per metric × cohort ×
    non-empty range, each computed on just that range's window."""
    import polars as pl

    from dashboard_api.services import common, series as series_mod

    df = pl.DataFrame(
        {
            "d": ["2026-01-01", "2026-01-02", "2026-06-01"],
            "user_id": ["u1", "u2", "u1"],
            "user_rtp": [0.5, 1.5, 2.0],
        }
    ).with_columns(pl.col("d").str.to_datetime())
    monkeypatch.setattr(series_mod, "load_lazy", lambda cfg, g: (df.lazy(), "d"))
    common._window_cache.clear()

    cfg = {"id": "rangetest", "stats_by_date": {}}
    _, out, missing = series_mod.load_series(
        cfg,
        "day",
        ["user_rtp"],
        ranges=[("2026-01-01", "2026-01-02"), (None, None), ("2026-06-01", "2026-06-30")],
    )
    common._window_cache.clear()
    assert missing == []
    assert [(s["range_index"], s["x"]) for s in out] == [
        (0, ["2026-01-01", "2026-01-02"]),
        (2, ["2026-06-01"]),
    ]
    assert out[0]["range_label"] == "2026-01-01 → 2026-01-02"


def test_lifecycle_group_schema_validation():
    import pytest
    from pydantic import ValidationError

    from dashboard_api.schemas.data import LifecycleGroup, SeriesRequest

    g = LifecycleGroup(label="new", start=0, end=3)
    assert g.end == 3
    assert LifecycleGroup(label="old", start=8).end is None  # open-ended
    with pytest.raises(ValidationError):
        LifecycleGroup(label="bad", start=5, end=2)  # inverted range
    with pytest.raises(ValidationError):
        LifecycleGroup(label="empty", start=5, end=5)  # end is exclusive → [5,5) is empty
    with pytest.raises(ValidationError):
        LifecycleGroup(label="neg", start=-1)

    req = SeriesRequest(config="c", metrics=["m"])
    assert req.lifecycle_groups is None  # absent for old clients → unchanged behavior


def test_config_detail_exposes_lifecycle_col():
    """The frontend shows the global lifecycle picker only when the config has a
    life_cycle_group cohort dimension."""
    from pathlib import Path

    from dashboard_api.services.configs import build_config_detail, load_raw_config

    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "dashboard")
    ss02 = load_raw_config(config_dir, "ss02")
    assert ss02 is not None and build_config_detail("ss02", ss02).lifecycle_col == "life_cycle_group"
    cluster = load_raw_config(config_dir, "ss03_user_cluster")
    assert cluster is not None and build_config_detail("ss03_user_cluster", cluster).lifecycle_col is None


def test_group_distribution_and_deepdive_requests_accept_filter():
    from dashboard_api.schemas.data import DeepdiveRequest, FilterOpts, GroupDistributionRequest

    req = GroupDistributionRequest(config="c", metric="m", ranges=[{"start": "2026-01-01", "end": "2026-01-02"}])
    assert req.filter == FilterOpts()  # defaults off, so old clients are unaffected
    dd = DeepdiveRequest(
        config="c",
        panel="derived",
        mode="histogram",
        metrics=["m"],
        ranges=[{"start": "2026-01-01", "end": "2026-01-02"}],
        filter={"enable": True, "min": 1, "max": 99},
    )
    assert dd.filter.enable and dd.filter.min == 1 and dd.filter.max == 99


def test_cached_group_values_single_flight(monkeypatch):
    """Concurrent cold misses on one (config, granularity) key run exactly one
    vocabulary scan; the other callers wait for it and share the result."""
    import threading
    import time as _time

    from dashboard_api.routers import data as data_router

    monkeypatch.setattr(data_router, "_GROUP_VALUES_CACHE", {})
    monkeypatch.setattr(data_router, "_GROUP_VALUES_LOADING", {})
    monkeypatch.setattr(data_router, "_GROUP_VALUES_REFRESHING", set())

    calls = []
    release = threading.Event()

    def slow_load(cfg, granularity):
        calls.append(granularity)
        release.wait(timeout=5)
        return {"user_group": ["a", "b"]}

    monkeypatch.setattr(data_router, "load_group_values", slow_load)

    results: list[dict] = []
    threads = [
        threading.Thread(target=lambda: results.append(data_router._cached_group_values("cfg", {}, "day")))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    _time.sleep(0.2)  # let every thread reach the cache before the scan finishes
    release.set()
    for t in threads:
        t.join(timeout=5)

    assert len(calls) == 1
    assert len(results) == 8 and all(r == {"user_group": ["a", "b"]} for r in results)


def test_cached_group_values_serves_stale_and_refreshes(monkeypatch):
    """An expired entry is served immediately (stale-while-revalidate) and one
    background refresh replaces it."""
    import time as _time

    from dashboard_api.routers import data as data_router

    key = ("cfg", "day")
    stale = {"user_group": ["old"]}
    monkeypatch.setattr(data_router, "_GROUP_VALUES_CACHE", {key: (_time.monotonic() - 4000.0, stale)})
    monkeypatch.setattr(data_router, "_GROUP_VALUES_LOADING", {})
    monkeypatch.setattr(data_router, "_GROUP_VALUES_REFRESHING", set())
    monkeypatch.setattr(data_router, "load_group_values", lambda cfg, granularity: {"user_group": ["new"]})

    assert data_router._cached_group_values("cfg", {}, "day") == stale  # served stale, no wait

    deadline = _time.monotonic() + 5
    while _time.monotonic() < deadline:
        with data_router._group_values_lock:
            current = data_router._GROUP_VALUES_CACHE[key][1]
        if current == {"user_group": ["new"]}:
            break
        _time.sleep(0.05)
    assert current == {"user_group": ["new"]}


def test_projection_columns_covers_metric_deps_and_collapse_components():
    """The window projection must carry each metric's user_* deps plus the
    numerator/denominator (and weights) collapse_user_rows recombines —
    without them a pruned frame silently drops ratio columns."""
    from dashboard_api.services.common import projection_columns

    cfg = {
        "id": "t",
        "stats_by_date": {
            "user_group_cols": ["ab_group", "mathtable"],
            "user_row_grain": ["ab_group", "mathtable"],
        },
    }
    available = {
        "activity_date",
        "user_id",
        "ab_group",
        "mathtable",
        "user_rtp",
        "user_total_payout",
        "user_total_bet",
        "user_num_bets",
        "unrelated_metric_col",
    }
    cols = projection_columns(cfg, ["rtp"], "activity_date", available)
    assert cols is not None
    assert {"activity_date", "user_id", "ab_group", "mathtable"} <= set(cols)
    assert {"user_total_payout", "user_total_bet"} <= set(cols)  # rtp's user_* deps
    assert "unrelated_metric_col" not in cols

    # A raw user_rtp request pulls the ratio's components so
    # collapse_user_rows can recompute (not drop) it on grain configs.
    ratio = projection_columns(cfg, ["user_rtp"], "activity_date", available)
    assert ratio is not None and {"user_rtp", "user_total_payout", "user_total_bet"} <= set(ratio)

    raw = projection_columns(cfg, ["unrelated_metric_col"], "activity_date", available)
    assert raw is not None and "unrelated_metric_col" in raw


def test_availability_cols_defaults_and_filters():
    from dashboard_api.services.common import availability_cols

    base = {"stats_by_date": {"user_group_cols": ["ab_group", "mathtable", "user_group"]}}
    assert availability_cols(base) == ["ab_group", "mathtable", "user_group"]

    scoped = {
        "stats_by_date": {
            "user_group_cols": ["ab_group", "mathtable", "user_group"],
            "availability_cols": ["mathtable", "not_a_group_col"],
        }
    }
    assert availability_cols(scoped) == ["mathtable"]


def test_collect_window_caches_per_column_set(monkeypatch):
    """Projected collects are cached per column tuple so different metric
    selections don't serve each other's narrower frames."""
    import polars as pl

    from dashboard_api.services import common

    monkeypatch.setattr(common, "_window_cache", type(common._window_cache)())
    lf = pl.LazyFrame(
        {
            "activity_date": ["2026-01-01", "2026-01-02"],
            "user_id": ["u1", "u2"],
            "user_rtp": [0.9, 1.1],
            "user_num_bets": [10, 20],
        }
    ).with_columns(pl.col("activity_date").str.to_datetime())
    cfg = {"id": "t"}
    from datetime import datetime

    start, end = datetime(2026, 1, 1), datetime(2026, 1, 3)
    narrow = common.collect_window(cfg, "day", lf, "activity_date", start, end, columns=["activity_date", "user_id"])
    wide = common.collect_window(cfg, "day", lf, "activity_date", start, end)
    assert set(narrow.columns) == {"activity_date", "user_id"}
    assert set(wide.columns) == {"activity_date", "user_id", "user_rtp", "user_num_bets"}
    assert len(common._window_cache) == 2


def test_collect_window_serves_subrange_from_covering_entry(monkeypatch):
    """A narrower request is sliced from a fresh cached wider window (same
    config/granularity, columns available) instead of collecting again."""
    from datetime import datetime

    import polars as pl

    from dashboard_api.services import common

    monkeypatch.setattr(common, "_window_cache", type(common._window_cache)())
    wide = pl.LazyFrame(
        {
            "activity_date": ["2026-01-01", "2026-01-15", "2026-02-10"],
            "user_id": ["u1", "u2", "u3"],
            "user_num_bets": [1, 2, 3],
        }
    ).with_columns(pl.col("activity_date").str.to_datetime())
    cfg = {"id": "t"}
    common.collect_window(cfg, "day", wide, "activity_date", datetime(2026, 1, 1), datetime(2026, 3, 1))
    assert len(common._window_cache) == 1

    # Collecting this LazyFrame would raise — proof the subrange never collects.
    poisoned = pl.LazyFrame({"activity_date": ["boom"]}).with_columns(
        pl.col("activity_date").str.to_datetime(strict=True)
    )
    sub = common.collect_window(cfg, "day", poisoned, "activity_date", datetime(2026, 1, 10), datetime(2026, 1, 31))
    assert sub["user_id"].to_list() == ["u2"]
    assert len(common._window_cache) == 1  # slice not re-cached

    projected = common.collect_window(
        cfg,
        "day",
        poisoned,
        "activity_date",
        datetime(2026, 1, 10),
        datetime(2026, 1, 31),
        columns=["activity_date", "user_num_bets"],
    )
    assert set(projected.columns) == {"activity_date", "user_num_bets"}


def test_collect_window_covering_entry_respects_columns_and_config(monkeypatch):
    """A projected cached frame must not serve requests needing columns it
    lacks (or full-column requests), and other configs never match."""
    from datetime import datetime

    import polars as pl

    from dashboard_api.services import common

    monkeypatch.setattr(common, "_window_cache", type(common._window_cache)())
    lf = pl.LazyFrame({"activity_date": ["2026-01-05"], "user_id": ["u1"], "user_num_bets": [7]}).with_columns(
        pl.col("activity_date").str.to_datetime()
    )
    cfg = {"id": "t"}
    common.collect_window(
        cfg,
        "day",
        lf,
        "activity_date",
        datetime(2026, 1, 1),
        datetime(2026, 2, 1),
        columns=["activity_date", "user_id"],
    )

    # Needs user_num_bets, which the cached projection lacks -> real collect.
    out = common.collect_window(
        cfg,
        "day",
        lf,
        "activity_date",
        datetime(2026, 1, 2),
        datetime(2026, 1, 31),
        columns=["activity_date", "user_num_bets"],
    )
    assert out["user_num_bets"].to_list() == [7]
    assert len(common._window_cache) == 2

    # Different config never matches even with an identical window.
    other = common.collect_window(
        {"id": "other"},
        "day",
        lf,
        "activity_date",
        datetime(2026, 1, 2),
        datetime(2026, 1, 31),
        columns=["activity_date", "user_id"],
    )
    assert other["user_id"].to_list() == ["u1"]
    assert len(common._window_cache) == 3


def test_bootstrap_ci_caps_resample_size_with_rescale():
    """Large per-user samples use the m-out-of-n bootstrap (resample size
    capped at 10k, deviations rescaled by sqrt(m/n)) — uncapped resampling on
    ss03-sized cohorts took seconds per Stats-by-Group cell, and an uncorrected
    cap would inflate the CI ~sqrt(n/m)x. Small samples keep the plain
    percentile bootstrap."""
    import math
    import time

    import numpy as np

    from bituslabs_ds.metrics.user_stats_aggregates import _BOOTSTRAP_MAX_SAMPLE, _bootstrap_ci

    rng = np.random.default_rng(0)
    arr = rng.exponential(2.0, 500_000)  # skewed, like bet amounts

    t0 = time.perf_counter()
    lo, hi = _bootstrap_ci(arr, n_boot=500)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5  # uncapped resampling of 500k values takes seconds

    # Width must match the n-sized sampling error (CLT half-width), not the
    # m-sized one (which is sqrt(n/m) ~ 7x wider here).
    mean = arr.mean()
    half = 1.959964 * arr.std(ddof=1) / np.sqrt(len(arr))
    assert lo < mean < hi
    assert 0.7 < (hi - lo) / (2 * half) < 1.4

    small = rng.exponential(2.0, 200)
    assert len(small) < _BOOTSTRAP_MAX_SAMPLE
    slo, shi = _bootstrap_ci(small, n_boot=500)
    assert slo < small.mean() < shi

    nan_lo, nan_hi = _bootstrap_ci(np.array([1.0]))
    assert math.isnan(nan_lo) and math.isnan(nan_hi)


def test_cached_response_single_flight_and_ttl(monkeypatch):
    """Identical concurrent requests compute once and share the result; a
    loader error propagates without being cached."""
    import threading

    from dashboard_api.services import common

    monkeypatch.setattr(common, "_response_cache", type(common._response_cache)())
    monkeypatch.setattr(common, "_response_loading", {})

    calls = []
    release = threading.Event()

    def compute():
        calls.append(1)
        release.wait(timeout=5)
        return {"answer": 42}

    results = []
    threads = [threading.Thread(target=lambda: results.append(common.cached_response("k", compute))) for _ in range(6)]
    for t in threads:
        t.start()
    import time as _time

    _time.sleep(0.2)
    release.set()
    for t in threads:
        t.join(timeout=5)
    assert len(calls) == 1 and len(results) == 6 and all(r == {"answer": 42} for r in results)
    assert common.cached_response("k", compute) == {"answer": 42}  # hit, no recompute
    assert len(calls) == 1

    def boom():
        raise RuntimeError("nope")

    import pytest

    with pytest.raises(RuntimeError):
        common.cached_response("err", boom)
    assert "err" not in common._response_cache  # errors are not cached
