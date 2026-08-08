"""Smoke tests for the dashboard_api FastAPI app (Phase 0 foundation).

These exercise app construction and the OpenAPI contract the frontend client is
generated from, without S3/network or an HTTP client. They run with just the
``main,dashboard_api`` dependency set (the Docker runtime), so they make a fast,
reliable CI gate for the new service.
"""

import math
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from bituslabs_ds import s3_utils
from bituslabs_ds.metrics.user_stats_aggregates import _BOOTSTRAP_MAX_SAMPLE, _bootstrap_ci
from dashboard_api.app import app
from dashboard_api.routers import data as data_router
from dashboard_api.routers.health import health
from dashboard_api.schemas.data import (
    DeepdiveRequest,
    FilterOpts,
    GroupDistributionRequest,
    LifecycleGroup,
    Series,
    SeriesRequest,
)
from dashboard_api.services import common, configs, series as series_mod
from dashboard_api.services.common import SeriesError, availability_cols, collect_window, projection_columns
from dashboard_api.services.configs import build_config_detail, load_raw_config
from dashboard_api.services.deepdive import _percentile_filter, _percentile_row_mask
from dashboard_api.services.group_distribution import _percentile_filter as _gd_filter


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
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "dashboard")
    cfg = load_raw_config(config_dir, "ss02")
    assert cfg is not None and cfg["id"] == "ss02"


def test_load_config_from_s3(monkeypatch, mock_s3_client):
    """config_dir may be an s3:// URI: configs are listed/parsed straight from S3 so a
    new dashboard_config-*.yaml needs only an upload, no image rebuild."""

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


def test_iter_cohorts_grain_collapses_user_rows(monkeypatch):
    """Two-column grain (ab_group × mathtable): each bet lives in exactly one
    row, so group sums are always right, but per-user stats must collapse the
    grain to one row per user whenever a grain dimension is unselected —
    summing additive components and recomputing ratios from the sums."""

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


def test_derived_group_cols_attached_by_load_lazy(tmp_path):
    """ss03's "ai_group" picker: a config-declared rollup of the stored
    ab_group labels (AI vs non-AI), attached at load time so pickers, series
    and deep-dive all see it as if it were a stored column."""
    df = pl.DataFrame(
        {
            "activity_date": ["2026-06-01"] * 4,
            "user_id": ["u1", "u2", "u3", "u4"],
            "ab_group": ["AI", "AB_TEST_A", "AB_TEST_B", "Default"],
            "user_num_bets": [1, 2, 3, 4],
        }
    )
    path = tmp_path / "daily_stats"
    path.mkdir()
    df.write_parquet(path / "part.parquet")
    cfg = {
        "id": "g",
        "stats_by_date": {
            "files": {"day": [str(path)]},
            "user_group_cols": ["ab_group", "ai_group"],
            "derived_group_cols": {"ai_group": {"source": "ab_group", "groups": {"AI": ["AI"]}, "default": "non-AI"}},
        },
    }
    lf, _ = common.load_lazy(cfg, "day")
    out = lf.collect().sort("user_id")
    assert out["ai_group"].to_list() == ["AI", "non-AI", "non-AI", "non-AI"]

    values = series_mod.load_group_values(cfg, "day")
    assert values["ai_group"] == ["AI", "non-AI"]
    assert values["ab_group"] == ["AB_TEST_A", "AB_TEST_B", "AI", "Default"]


def test_attach_derived_group_cols_guards():
    df = pl.DataFrame({"ab_group": ["AI", "Default"], "ai_group": ["stored", "stored"]})
    cfg = {
        "id": "g",
        "stats_by_date": {
            "derived_group_cols": {
                "ai_group": {"source": "ab_group", "groups": {"AI": ["AI"]}, "default": "non-AI"},
                "other": {"source": "missing_col", "groups": {"X": ["x"]}},
            }
        },
    }
    out = common.attach_derived_group_cols(cfg, df.lazy()).collect()
    assert out["ai_group"].to_list() == ["stored", "stored"]  # never overwrites a real column
    assert "other" not in out.columns  # missing source is skipped

    # No default: unmatched values stay null (they only ever match 'all').
    cfg2 = {
        "id": "g",
        "stats_by_date": {"derived_group_cols": {"ai_group": {"source": "ab_group", "groups": {"AI": ["AI"]}}}},
    }
    out2 = common.attach_derived_group_cols(cfg2, df.drop("ai_group").lazy()).collect()
    assert out2["ai_group"].to_list() == ["AI", None]


def test_row_filters_applied_by_load_lazy(tmp_path):
    """Cohort-scoped configs (e.g. the SS03 AI dashboard): stats_by_date.filters
    keeps only the declared slice of the parquet, so every endpoint — including
    the 'all' cohort — describes just that slice."""
    df = pl.DataFrame(
        {
            "activity_date": ["2026-06-01"] * 4,
            "user_id": ["u1", "u2", "u3", "u4"],
            "ab_group": ["AI", "AB_TEST_A", "AI", "Default"],
            "user_num_bets": [1, 2, 3, 4],
        }
    )
    path = tmp_path / "daily_stats"
    path.mkdir()
    df.write_parquet(path / "part.parquet")
    cfg = {
        "id": "g",
        "stats_by_date": {"files": {"day": [str(path)]}, "filters": {"ab_group": ["AI"]}},
    }
    lf, _ = common.load_lazy(cfg, "day")
    out = lf.collect()
    assert sorted(out["user_id"].to_list()) == ["u1", "u3"]

    # A scalar value works like a one-element list.
    cfg["stats_by_date"]["filters"] = {"ab_group": "Default"}
    assert common.load_lazy(cfg, "day")[0].collect()["user_id"].to_list() == ["u4"]

    # A filter on a column the data doesn't have must fail loudly — skipping it
    # would serve the whole game's rows under a config that promises a slice.
    cfg["stats_by_date"]["filters"] = {"missing_col": ["AI"]}
    with pytest.raises(common.SeriesError, match="missing_col"):
        common.load_lazy(cfg, "day")

    # Filters see derived group columns (attached before filtering applies).
    cfg["stats_by_date"]["filters"] = {"ai_group": ["non-AI"]}
    cfg["stats_by_date"]["derived_group_cols"] = {
        "ai_group": {"source": "ab_group", "groups": {"AI": ["AI"]}, "default": "non-AI"}
    }
    out = common.load_lazy(cfg, "day")[0].collect()
    assert sorted(out["user_id"].to_list()) == ["u2", "u4"]


def _combo_cfg(path, filters=None, **combo_opts):
    sd = {
        "files": {"day": [str(path)]},
        "user_group_cols": ["mathtable_combo"],
        "user_row_grain": ["mathtable"],
        "combo_group_cols": {
            "mathtable_combo": {
                "source": "mathtable",
                "weight": "user_num_bets",
                "strip_prefix": "normal_",
                "label_prefix": "ai",
                **combo_opts,
            }
        },
    }
    if filters:
        sd["filters"] = filters
    return {"id": "g", "stats_by_date": sd}


def test_combo_group_cols_label_per_user_day(tmp_path):
    """ss03-AI's mathtable_combo picker: each user-day is labeled with the
    mathtables the user played that day, dominant (most bets) first, ties
    alphabetical; every row of the user-day carries the same label."""
    df = pl.DataFrame(
        {
            "activity_date": ["2026-06-01"] * 4 + ["2026-06-02"],
            "user_id": ["u1", "u1", "u1", "u2", "u1"],
            "ab_group": ["AI"] * 5,
            # u1 day1: shi dominates zero (12 > 10 bets, summed across
            # bet_levels); u2 plays a single table; u1 day2 relabels.
            "mathtable": ["normal_zero", "normal_shi", "normal_shi", "normal_ichi", "normal_zero"],
            "bet_level": [1.0, 1.0, 2.0, 1.0, 1.0],
            "user_num_bets": [10, 5, 7, 3, 4],
        }
    )
    path = tmp_path / "daily_stats"
    path.mkdir()
    df.write_parquet(path / "part.parquet")
    out = common.load_lazy(_combo_cfg(path), "day")[0].collect()
    by = {(r["activity_date"], r["user_id"], r["mathtable"]): r["mathtable_combo"] for r in out.iter_rows(named=True)}
    assert by[("2026-06-01", "u1", "normal_zero")] == "ai_shi_zero"
    assert by[("2026-06-01", "u1", "normal_shi")] == "ai_shi_zero"
    assert by[("2026-06-01", "u2", "normal_ichi")] == "ai_ichi"
    assert by[("2026-06-02", "u1", "normal_zero")] == "ai_zero"

    # Ties order alphabetically: zero(10) vs ichi(10) -> ichi first.
    tie = pl.DataFrame(
        {
            "activity_date": ["2026-06-01"] * 2,
            "user_id": ["u3", "u3"],
            "mathtable": ["normal_zero", "normal_ichi"],
            "bet_level": [1.0, 1.0],
            "user_num_bets": [10, 10],
        }
    )
    tie.write_parquet(path / "part.parquet")
    out = common.load_lazy(_combo_cfg(path), "day")[0].collect()
    assert out["mathtable_combo"].to_list() == ["ai_ichi_zero"] * 2


def test_combo_group_cols_order_tiebreak_and_max_tables(tmp_path):
    """Equal bet counts order by which mathtable the user played FIRST (the
    ETL's user_first_spin_id), and max_tables caps the label at the first N
    tables with a trailing '+' when the combination is longer."""
    path = tmp_path / "daily_stats"
    path.mkdir()
    # u1: ichi(10) ties zero(10) but was played first -> ichi leads, and with
    # zero's earlier alphabetical rank the order column must be what decides.
    tie = pl.DataFrame(
        {
            "activity_date": ["2026-06-01"] * 2,
            "user_id": ["u1", "u1"],
            "mathtable": ["normal_zero", "normal_ichi"],
            "bet_level": [1.0, 1.0],
            "user_num_bets": [10, 10],
            "user_first_spin_id": [200, 100],
        }
    )
    tie.write_parquet(path / "part.parquet")
    out = common.load_lazy(_combo_cfg(path, order="user_first_spin_id"), "day")[0].collect()
    assert out["mathtable_combo"].to_list() == ["ai_ichi_zero"] * 2
    # Order column configured but absent from the data: alphabetical fallback.
    tie.drop("user_first_spin_id").write_parquet(path / "part.parquet")
    out = common.load_lazy(_combo_cfg(path, order="user_first_spin_id"), "day")[0].collect()
    assert out["mathtable_combo"].to_list() == ["ai_ichi_zero"] * 2  # alphabetical happens to agree
    # Five tables, distinct weights: label keeps the top 4 and flags the rest.
    five = pl.DataFrame(
        {
            "activity_date": ["2026-06-01"] * 5,
            "user_id": ["u2"] * 5,
            "mathtable": [f"normal_m{i}" for i in range(5)],
            "bet_level": [1.0] * 5,
            "user_num_bets": [50, 40, 30, 20, 10],
            "user_first_spin_id": list(range(5)),
        }
    )
    five.write_parquet(path / "part.parquet")
    cfg = _combo_cfg(path, separator="-", max_tables=4)
    out = common.load_lazy(cfg, "day")[0].collect()
    assert out["mathtable_combo"].to_list() == ["ai_m0-m1-m2-m3+"] * 5
    # Exactly max_tables tables merges into the SAME '+' label as longer
    # days with the same leading four, so 4 and 4+ aren't split.
    four = five.head(4)
    four.write_parquet(path / "part.parquet")
    out = common.load_lazy(cfg, "day")[0].collect()
    assert out["mathtable_combo"].to_list() == ["ai_m0-m1-m2-m3+"] * 4
    # Below the cap: plain label, no '+'.
    five.head(3).write_parquet(path / "part.parquet")
    out = common.load_lazy(cfg, "day")[0].collect()
    assert out["mathtable_combo"].to_list() == ["ai_m0-m1-m2"] * 3


def test_combo_group_cols_min_user_days_folds_rare_labels(tmp_path):
    """Combinations with fewer than min_user_days (user, day) samples in the
    full daily history fold into other_label, so the picker only lists
    well-populated combo groups; the vocabulary is cached per config."""
    common._combo_vocab_cache.clear()
    path = tmp_path / "daily_stats"
    path.mkdir()
    df = pl.DataFrame(
        {
            # normal_a alone: 2 user-day samples (u1 d1, u2 d1); normal_b: 1.
            "activity_date": ["2026-06-01", "2026-06-01", "2026-06-02"],
            "user_id": ["u1", "u2", "u3"],
            "mathtable": ["normal_a", "normal_a", "normal_b"],
            "bet_level": [1.0, 1.0, 1.0],
            "user_num_bets": [5, 5, 5],
        }
    )
    df.write_parquet(path / "part.parquet")
    cfg = _combo_cfg(path, min_user_days=2, other_label="ai_other")
    cfg["id"] = "g-vocab"
    out = common.load_lazy(cfg, "day")[0].collect().sort("user_id")
    assert out["mathtable_combo"].to_list() == ["ai_a", "ai_a", "ai_other"]
    values = series_mod.load_group_values(cfg, "day")
    assert values["mathtable_combo"] == ["ai_a", "ai_other"]

    # other_label defaults to '<label_prefix>_other' when not configured.
    common._combo_vocab_cache.clear()
    cfg2 = _combo_cfg(path, min_user_days=2)
    cfg2["id"] = "g-vocab-2"
    out2 = common.load_lazy(cfg2, "day")[0].collect().sort("user_id")
    assert out2["mathtable_combo"].to_list() == ["ai_a", "ai_a", "ai_other"]


def test_combo_group_cols_after_filters_and_guards(tmp_path):
    """The combination counts only the config's filtered slice (an AI-only
    config must not fold a user's Default-group rows into their combo), and
    missing inputs / already-stored columns are left untouched."""
    df = pl.DataFrame(
        {
            "activity_date": ["2026-06-01"] * 2,
            "user_id": ["u1", "u1"],
            "ab_group": ["AI", "Default"],
            "mathtable": ["normal_zero", "normal_shi"],
            "bet_level": [1.0, 1.0],
            "user_num_bets": [5, 50],
        }
    )
    path = tmp_path / "daily_stats"
    path.mkdir()
    df.write_parquet(path / "part.parquet")
    out = common.load_lazy(_combo_cfg(path, filters={"ab_group": ["AI"]}), "day")[0].collect()
    assert out["mathtable_combo"].to_list() == ["ai_zero"]  # Default row excluded entirely

    # A stored column with the same name wins; a missing weight column skips.
    stored = df.with_columns(pl.lit("stored").alias("mathtable_combo"))
    cfg = _combo_cfg(path)
    assert common.attach_combo_group_cols(cfg, stored.lazy(), "activity_date").collect()[
        "mathtable_combo"
    ].to_list() == ["stored", "stored"]
    out2 = common.attach_combo_group_cols(cfg, df.drop("user_num_bets").lazy(), "activity_date").collect()
    assert "mathtable_combo" not in out2.columns


def test_combo_group_cols_keep_period_pruning(tmp_path):
    """The window-cache period predicate must still prune parquet paths under
    the combo window expressions (period is in the partition keys precisely
    so polars can push it down) — otherwise every request re-reads the full
    history and the ss01-scale latency fix regresses."""
    root = tmp_path / "daily_stats"
    for d in ("2026-06-01", "2026-06-02", "2026-06-03"):
        (root / f"period={d}").mkdir(parents=True)
        pl.DataFrame(
            {
                "activity_date": [d],
                "user_id": ["u1"],
                "mathtable": ["normal_zero"],
                "bet_level": [1.0],
                "user_num_bets": [1],
            }
        ).write_parquet(root / f"period={d}" / "part.parquet")
    lf, _ = common.load_lazy(_combo_cfg(root), "day")
    plan = common._prune_periods(lf, "day", "2026-06-02", "2026-06-02").explain()
    assert "other sources" not in plan  # 1 of 3 files scanned, not all


def test_iter_cohorts_derived_group_collapses_grain(monkeypatch):
    """Selecting ai_group='non-AI' with ab_group unselected must collapse the
    per-(user, ab_group) grain rows back to one row per user, so per-user
    stats count each user once across the combined subgroups."""
    df = pl.DataFrame(
        {
            "d": ["2026-06-01"] * 4,
            "user_id": ["u1", "u1", "u2", "u2"],
            "ab_group": ["AB_TEST_A", "Default", "AI", "AB_TEST_B"],
            "user_num_bets": [10, 30, 5, 7],
            "user_total_bet": [100.0, 300.0, 50.0, 70.0],
        }
    ).with_columns(pl.col("d").str.to_datetime())
    cfg = {
        "id": "g",
        "stats_by_date": {
            "user_group_cols": ["ab_group", "ai_group"],
            "user_row_grain": ["ab_group"],
            "derived_group_cols": {"ai_group": {"source": "ab_group", "groups": {"AI": ["AI"]}, "default": "non-AI"}},
        },
    }
    df = common.attach_derived_group_cols(cfg, df.lazy()).collect()

    out = dict(common.iter_cohorts(cfg, df, {"ai_group": ["AI", "non-AI"]}, date_col="d"))
    assert set(out) == {"AI", "non-AI"}
    non_ai = out["non-AI"].sort("user_id")
    assert non_ai.height == 2  # u1's two subgroup rows collapsed; u2 keeps only AB_TEST_B
    assert non_ai["user_num_bets"].to_list() == [40, 7]
    assert out["AI"].height == 1 and out["AI"]["user_num_bets"][0] == 5

    # Pinning both dimensions intersects them: AI ∩ non-AI is empty.
    crossed = dict(common.iter_cohorts(cfg, df, {"ab_group": ["AI"], "ai_group": ["non-AI"]}, date_col="d"))
    assert crossed["AI | non-AI"].height == 0


def test_iter_cohorts_range_groups_bucket_by_value(monkeypatch):
    """The range-group dimension (e.g. fish_value) buckets rows by INCLUSIVE
    [min, max] ranges from the request; a pinned range still collapses the
    grain (one range spans many stored values), and 'all' is no filter."""

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

    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "dashboard")
    ss02 = load_raw_config(config_dir, "ss02")
    assert ss02 is not None and build_config_detail("ss02", ss02).lifecycle_col == "life_cycle_group"
    cluster = load_raw_config(config_dir, "ss03_user_cluster")
    assert cluster is not None and build_config_detail("ss03_user_cluster", cluster).lifecycle_col is None


def test_group_distribution_and_deepdive_requests_accept_filter():
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

    calls = []
    release = threading.Event()

    def slow_load(cfg, granularity):
        calls.append(granularity)
        release.wait(timeout=5)
        return {"ab_test_group": ["a", "b"]}

    monkeypatch.setattr(data_router, "_GROUP_VALUES_CACHE", {})
    monkeypatch.setattr(data_router, "_GROUP_VALUES_LOADING", {})
    monkeypatch.setattr(data_router, "_GROUP_VALUES_REFRESHING", set())
    monkeypatch.setattr(data_router, "load_group_values", slow_load)

    results: list[dict] = []
    threads = [
        threading.Thread(target=lambda: results.append(data_router._cached_group_values("cfg", {}, "day")))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    time.sleep(0.2)  # let every thread reach the cache before the scan finishes
    release.set()
    for t in threads:
        t.join(timeout=5)

    assert len(calls) == 1
    assert len(results) == 8 and all(r == {"ab_test_group": ["a", "b"]} for r in results)


def test_cached_group_values_serves_stale_and_refreshes(monkeypatch):
    """An expired entry is served immediately (stale-while-revalidate) and one
    background refresh replaces it."""

    key = ("cfg", "day")
    stale = {"ab_test_group": ["old"]}
    monkeypatch.setattr(data_router, "_GROUP_VALUES_CACHE", {key: (time.monotonic() - 4000.0, stale)})
    monkeypatch.setattr(data_router, "_GROUP_VALUES_LOADING", {})
    monkeypatch.setattr(data_router, "_GROUP_VALUES_REFRESHING", set())
    monkeypatch.setattr(data_router, "load_group_values", lambda cfg, granularity: {"ab_test_group": ["new"]})

    assert data_router._cached_group_values("cfg", {}, "day") == stale  # served stale, no wait

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with data_router._group_values_lock:
            current = data_router._GROUP_VALUES_CACHE[key][1]
        if current == {"ab_test_group": ["new"]}:
            break
        time.sleep(0.05)
    assert current == {"ab_test_group": ["new"]}


def test_projection_columns_covers_metric_deps_and_collapse_components():
    """The window projection must carry each metric's user_* deps plus the
    numerator/denominator (and weights) collapse_user_rows recombines —
    without them a pruned frame silently drops ratio columns."""

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
    base = {"stats_by_date": {"user_group_cols": ["ab_group", "mathtable", "bet_level"]}}
    assert availability_cols(base) == ["ab_group", "mathtable", "bet_level"]

    scoped = {
        "stats_by_date": {
            "user_group_cols": ["ab_group", "mathtable", "bet_level"],
            "availability_cols": ["mathtable", "not_a_group_col"],
        }
    }
    assert availability_cols(scoped) == ["mathtable"]


def test_collect_window_caches_per_column_set(monkeypatch):
    """Projected collects are cached per column tuple so different metric
    selections don't serve each other's narrower frames."""

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

    start, end = datetime(2026, 1, 1), datetime(2026, 1, 3)
    narrow = common.collect_window(cfg, "day", lf, "activity_date", start, end, columns=["activity_date", "user_id"])
    wide = common.collect_window(cfg, "day", lf, "activity_date", start, end)
    assert set(narrow.columns) == {"activity_date", "user_id"}
    assert set(wide.columns) == {"activity_date", "user_id", "user_rtp", "user_num_bets"}
    assert len(common._window_cache) == 2


def test_collect_window_serves_subrange_from_covering_entry(monkeypatch):
    """A narrower request is sliced from a fresh cached wider window (same
    config/granularity, columns available) instead of collecting again."""

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

    time.sleep(0.2)
    release.set()
    for t in threads:
        t.join(timeout=5)
    assert len(calls) == 1 and len(results) == 6 and all(r == {"answer": 42} for r in results)
    assert common.cached_response("k", compute) == {"answer": 42}  # hit, no recompute
    assert len(calls) == 1

    def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        common.cached_response("err", boom)
    assert "err" not in common._response_cache  # errors are not cached


def test_iter_cohorts_explicit_empty_selection_yields_no_cohorts():
    """A column present with an EMPTY selection means the caller unselected
    everything: no cohorts, nothing plotted. Absent columns still mean 'all'."""

    df = pl.DataFrame({"d": ["2026-06-01"], "user_id": ["u1"], "ai_group": ["A"]}).with_columns(
        pl.col("d").str.to_datetime()
    )
    cfg = {"id": "t", "stats_by_date": {"user_group_cols": ["ai_group"]}}

    assert dict(common.iter_cohorts(cfg, df, {"ai_group": []}, date_col="d")) == {}
    assert list(dict(common.iter_cohorts(cfg, df, {}, date_col="d"))) == ["all"]


def test_empty_results_are_not_cached(monkeypatch):
    """An empty collect can be a read racing the ETL's partition rewrite —
    caching it would pin 'no data' on every panel for the TTL. Empty frames
    and empty responses are returned but recomputed next request."""

    monkeypatch.setattr(common, "_window_cache", type(common._window_cache)())
    empty = pl.LazyFrame({"activity_date": [], "user_id": []}).with_columns(pl.col("activity_date").cast(pl.Datetime))
    out = common.collect_window({"id": "t"}, "day", empty, "activity_date", datetime(2026, 1, 1), datetime(2026, 2, 1))
    assert out.is_empty()
    assert len(common._window_cache) == 0  # not cached

    monkeypatch.setattr(common, "_response_cache", type(common._response_cache)())
    monkeypatch.setattr(common, "_response_loading", {})
    calls = []

    def compute():
        calls.append(1)
        return ("date", [], ["metric"])  # empty series

    for _ in range(2):
        assert common.cached_response("k", compute, should_cache=lambda v: bool(v[1])) == ("date", [], ["metric"])
    assert len(calls) == 2  # recomputed, not served from cache


def test_load_lazy_hive_scan_and_period_pruning(tmp_path, monkeypatch):
    """Sources with period= layouts load through ONE hive-aware scan (not one
    scan + schema fetch per file — request latency scaled with history length
    on long-lived games like ss01), and collect_window prunes partitions by
    path before reading them. Flat layouts pass through untouched."""

    hive_root = tmp_path / "daily_stats"
    for day, uid_ in [("2026-07-01", "u1"), ("2026-07-02", "u2"), ("2026-07-03", "u3")]:
        d = hive_root / f"period={day}"
        d.mkdir(parents=True)
        pl.DataFrame({"activity_date": [day], "user_id": [uid_], "user_num_bets": [1]}).with_columns(
            pl.col("activity_date").str.to_datetime()
        ).write_parquet(d / "part.parquet")

    cfg = {"id": "hive", "stats_by_date": {"files": {"day": [str(hive_root)]}, "date_col": "activity_date"}}
    lf, date_col = common.load_lazy(cfg, "day")
    assert date_col == "activity_date"
    assert "period" in lf.collect_schema().names()  # hive column materialized

    monkeypatch.setattr(common, "_window_cache", type(common._window_cache)())
    out = common.collect_window(cfg, "day", lf, date_col, datetime(2026, 7, 2), datetime(2026, 7, 3))
    assert sorted(out["user_id"].to_list()) == ["u2", "u3"]  # pruned + filtered correctly

    flat_root = tmp_path / "flat_stats"
    flat_root.mkdir()
    pl.DataFrame({"activity_date": ["2026-07-01"], "user_id": ["u9"]}).with_columns(
        pl.col("activity_date").str.to_datetime()
    ).write_parquet(flat_root / "all.parquet")
    flat_cfg = {"id": "flat", "stats_by_date": {"files": {"day": [str(flat_root)]}, "date_col": "activity_date"}}
    flat_lf, _ = common.load_lazy(flat_cfg, "day")
    assert "period" not in flat_lf.collect_schema().names()
    flat_out = common.collect_window(
        flat_cfg, "day", flat_lf, "activity_date", datetime(2026, 7, 1), datetime(2026, 7, 2)
    )
    assert flat_out["user_id"].to_list() == ["u9"]
