"""
FTUE event-timing analysis for the FM01 first-session dataset.

Usage:
    python analysis_ftue_events.py [--bin-seconds 30] [--high-fish-value 500]

Reads the parquet dataset produced by etl_game_stats_ftue.py (one row per
bullet in each new user's first session, with attached CNY transactions) and
generates an interactive HTML report. For each user it finds the timing of
these milestone events, measured from both account-created and first-bet time:

    - first bet            (first bullet fired)
    - first kill           (first bullet with payout > 0)
    - first high-value kill (first kill with fish_value >= threshold)
    - stop play            (last bullet of the first session)
    - first deposit        (first deposit transaction, amount > 0)
    - second deposit       (second deposit transaction, amount > 0)
    - first withdrawal     (first withdrawal transaction, amount > 0)
    - change device        (first bullet whose device_type differs from the first bullet)
    - change IP            (first bullet whose ip differs from the first bullet)

Each user is assigned the strategy_name of their first bullet (the strategy at
session start). The report is a single interactive figure: pick the time
origin, up to two strategy groups to compare (Group A solid, Group B dashed),
and which metrics to draw; the summary table below rebuilds from the selection.

A second section plots per-user bet-volume metrics as time series over each
user's first ``--window-minutes`` (default 40) from the selected origin: for
every bin_seconds bin, the cross-user mean with a 95% CI band, for average bets
per user, average total bet per user, share of users with at least one bet, and
the per-user share of consecutive bet-to-bet changes that go up / down. A
40-minute summary table sits below it. The event-timing curves also default-zoom
to this window.
"""

import argparse
import json
import os
from datetime import datetime
from typing import cast

import awswrangler as wr
import numpy as np
import pandas as pd

from bituslabs_ds.config import DEFAULT_ETL_OUTPUT, LOCAL_ROOT, setup_logging

DEFAULT_INPUT = f"{DEFAULT_ETL_OUTPUT}/jobs/output_fish_hunter/ftue_first_session/"
DEFAULT_OUTPUT = f"{LOCAL_ROOT}/jobs/output_fish_hunter/ftue_analysis_report.html"
DEFAULT_MD_OUTPUT = f"{LOCAL_ROOT}/jobs/output_fish_hunter/ftue_bet_behavior_zh.md"

COLUMNS = [
    "user_id",
    "account_created",
    "event_timestamp",
    "bullet_id",
    "bet",
    "payout",
    "fish_value",
    "device_type",
    "ip",
    "strategy_name",
    "transaction_type",
    "transaction_amount",
    "transaction_processed_at",
]

EVENT_LABELS = {
    "first_bet": "First bet",
    "first_kill": "First kill (payout > 0)",
    "first_high_kill": "First high-value kill",
    "stop_play": "Stop play (last bet)",
    "first_deposit": "First deposit",
    "second_deposit": "Second deposit",
    "first_withdrawal": "First withdrawal",
    "change_device": "Change device",
    "change_ip": "Change IP",
}

# Lines are colored by strategy group (stable per group index); metric is
# encoded by dash style (or solid when a single metric is selected). High-
# contrast ColorBrewer Set1 palette so groups stay distinct.
GROUP_PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]


def _group_color(i: int) -> str:
    return GROUP_PALETTE[i % len(GROUP_PALETTE)]


# (origin column, axis label, events visible by default in that figure)
ORIGINS = [
    ("account_created", "account created", ("first_bet", "stop_play")),
    ("first_bet", "first bet", ("first_kill", "stop_play")),
]


def load_data(input_path: str) -> pd.DataFrame:
    df = wr.s3.read_parquet(path=input_path, dataset=True, columns=COLUMNS)
    print(f"Loaded {len(df)} bullet rows for {df['user_id'].nunique()} users from {input_path}")
    return df


def compute_user_events(df: pd.DataFrame, high_fish_value: float) -> pd.DataFrame:
    """One row per user with the timestamp of each milestone event (NaT if never reached).

    Bullets fan out across attached transactions (a bullet with N transactions
    appears N times), but every per-user metric here is a min/max/first over
    timestamps, which is idempotent under that duplication. Each transaction
    attaches to exactly one bullet, so deposit/withdrawal ordering is unaffected.
    """
    df = df.sort_values("event_timestamp")
    g = df.groupby("user_id")

    events = cast(
        pd.DataFrame,
        g.agg(
            account_created=("account_created", "first"),
            first_bet=("event_timestamp", "min"),
            stop_play=("event_timestamp", "max"),
            strategy_group=("strategy_name", "first"),
        ),
    )

    kills = df.loc[df["payout"] > 0]
    high_kills = kills.loc[kills["fish_value"] >= high_fish_value]
    events["first_kill"] = kills.groupby("user_id")["event_timestamp"].min()
    events["first_high_kill"] = high_kills.groupby("user_id")["event_timestamp"].min()

    # Device / IP change: earliest bullet whose value differs from the user's
    # first bullet. notna guard avoids treating a missing value as a change.
    first_device = g["device_type"].transform("first")
    first_ip = g["ip"].transform("first")
    dev_chg = df.loc[df["device_type"].notna() & (df["device_type"] != first_device)]
    ip_chg = df.loc[df["ip"].notna() & (df["ip"] != first_ip)]
    events["change_device"] = dev_chg.groupby("user_id")["event_timestamp"].min()
    events["change_ip"] = ip_chg.groupby("user_id")["event_timestamp"].min()

    # Deposits / withdrawals with amount > 0, ordered by processed_at.
    txn = df.loc[df["transaction_type"].notna() & (df["transaction_amount"] > 0)]
    deposits = (
        txn.loc[txn["transaction_type"] == "deposit", ["user_id", "transaction_processed_at"]]
        .drop_duplicates()
        .sort_values(["user_id", "transaction_processed_at"])
    )
    dep_rank = deposits.groupby("user_id").cumcount()
    events["first_deposit"] = deposits.loc[dep_rank == 0].set_index("user_id")["transaction_processed_at"]
    events["second_deposit"] = deposits.loc[dep_rank == 1].set_index("user_id")["transaction_processed_at"]
    withdrawals = txn.loc[txn["transaction_type"] == "withdrawal"]
    events["first_withdrawal"] = withdrawals.groupby("user_id")["transaction_processed_at"].min()

    return events.reset_index()


DEFAULT_WINDOW_MINUTES = 40


def _mean_ci(x: pd.Series) -> dict:
    """Cross-user mean of a per-user metric with a 95% normal-approx CI (mean +- 1.96*SEM)."""
    x = x.dropna()
    n = int(len(x))
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    mean = float(x.mean())
    half = 1.96 * float(x.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    return {"mean": round(mean, 3), "lo": round(mean - half, 3), "hi": round(mean + half, 3), "n": n}


def _prop_ci(k: int, n: int) -> dict:
    """Proportion k/n with a 95% normal-approx (Wald) CI, clamped to [0, 1]."""
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "reached": 0, "n": 0}
    p = k / n
    half = 1.96 * np.sqrt(p * (1 - p) / n)
    return {
        "mean": round(p, 4),
        "lo": round(max(0.0, p - half), 4),
        "hi": round(min(1.0, p + half), 4),
        "reached": int(k),
        "n": int(n),
    }


def compute_user_bet_window(df: pd.DataFrame, window_seconds: int) -> dict:
    """Per-user bet-volume metrics within the first ``window_seconds`` of each origin.

    Returns ``{origin_label: per_user_df}`` (per_user_df indexed by user_id) with
    columns: strategy_group, n_bets, total_bet, has_bet, ratio_up, ratio_down.
    ``ratio_up``/``ratio_down`` are the fraction of a user's consecutive
    bet-to-bet deltas that are positive / negative (NaN for users with < 2 bets
    in the window, i.e. no delta to evaluate).

    Bullets are de-duplicated on (user_id, bullet_id) first because the
    transaction fan-out in the source dataset repeats a bullet row once per
    attached transaction; counting raw rows would over-count bets.
    """
    bullets = df.dropna(subset=["bullet_id"]).drop_duplicates(["user_id", "bullet_id"])
    base = cast(
        pd.DataFrame,
        bullets.groupby("user_id").agg(
            account_created=("account_created", "first"),
            first_bet=("event_timestamp", "min"),
            strategy_group=("strategy_name", "first"),
        ),
    )
    window = pd.Timedelta(seconds=window_seconds)
    cols = ["user_id", "event_timestamp", "bullet_id", "bet"]
    out: dict = {}
    for origin_col, origin_label, _ in ORIGINS:
        origin_time = cast(pd.Series, base[origin_col]).rename("origin_time")
        m = bullets[cols].merge(origin_time, left_on="user_id", right_index=True)
        win = m.loc[(m["event_timestamp"] >= m["origin_time"]) & (m["event_timestamp"] <= m["origin_time"] + window)]
        win = win.sort_values(["user_id", "event_timestamp", "bullet_id"])

        agg = win.groupby("user_id").agg(n_bets=("bet", "size"), total_bet=("bet", "sum"))
        delta = win.groupby("user_id")["bet"].diff()  # NaN on each user's first bet
        uid = win["user_id"]
        up = delta.gt(0).groupby(uid).sum()
        dn = delta.lt(0).groupby(uid).sum()
        n_delta = win.groupby("user_id").size() - 1
        n_delta = n_delta.where(n_delta > 0, np.nan)  # < 2 bets -> no delta -> NaN ratio

        per_user = base[["strategy_group"]].join(agg)
        per_user["n_bets"] = per_user["n_bets"].fillna(0).astype(int)
        per_user["total_bet"] = per_user["total_bet"].fillna(0.0)
        per_user["has_bet"] = per_user["n_bets"] >= 1
        per_user["ratio_up"] = up / n_delta
        per_user["ratio_down"] = dn / n_delta
        # whether the user ever raised / lowered their bet in the window (>=1 delta)
        per_user["any_up"] = up.reindex(per_user.index).fillna(0) > 0
        per_user["any_down"] = dn.reindex(per_user.index).fillna(0) > 0
        out[origin_label] = per_user
    return out


# Per-bin bet metrics. "Avg ... / active user" divides by the users who actually
# bet in that bin (not the whole 40-min cohort); the "Total ..." metrics are the
# raw per-bin sums. Grouped for the checkbox layout; flat dict drives everything else.
BET_METRIC_GROUPS = [
    ("Bet volume", ["sum_n_bets", "n_bets", "sum_total_bet", "total_bet"]),
    ("Active users", ["n_with_bet", "with_bet"]),
    ("Per-user bet change", ["ratio_up", "ratio_down"]),
    ("Users changing bet", ["n_up_users", "r_up_users", "n_down_users", "r_down_users"]),
]
BET_METRICS = {
    "sum_n_bets": "Total bets",
    "n_bets": "Avg bets / active user",
    "sum_total_bet": "Total bet amount",
    "total_bet": "Avg bet amount / active user",
    "n_with_bet": "Users with a bet (n)",
    "with_bet": "Users with a bet (%)",
    "ratio_up": "Bet-increase ratio (per-user deltas)",
    "ratio_down": "Bet-decrease ratio (per-user deltas)",
    "n_up_users": "Users increasing bet (n)",
    "r_up_users": "Users increasing bet (%)",
    "n_down_users": "Users decreasing bet (n)",
    "r_down_users": "Users decreasing bet (%)",
}


def bet_window_config(df: pd.DataFrame, window_seconds: int) -> dict:
    """Aggregate the per-user bet-window metrics to {origin: {group: {metric: mean+CI}}}."""
    per_user_by_origin = compute_user_bet_window(df, window_seconds)
    out: dict = {}
    for _, origin_label, _ in ORIGINS:
        pu = per_user_by_origin[origin_label]
        groups = [("All users", pu)]
        for grp in sorted(pu["strategy_group"].dropna().unique()):
            groups.append((str(grp), pu.loc[pu["strategy_group"] == grp]))
        gdict: dict = {}
        for label, sub in groups:
            n_users = int(len(sub))
            active = sub.loc[sub["has_bet"]]  # per-active-user means exclude users with no bet in the window
            gdict[label] = {
                "n_users": n_users,
                "n_bets": _mean_ci(active["n_bets"]),
                "total_bet": _mean_ci(active["total_bet"]),
                "with_bet": _prop_ci(int(sub["has_bet"].sum()), n_users),
                "ratio_up": _mean_ci(sub["ratio_up"]),
                "ratio_down": _mean_ci(sub["ratio_down"]),
                "users_up": _prop_ci(int(sub["any_up"].sum()), n_users),
                "users_down": _prop_ci(int(sub["any_down"].sum()), n_users),
            }
        out[origin_label] = gdict
    return out


DEFAULT_BET_METRICS = ("with_bet",)


def _round_list(arr: np.ndarray, nd: int = 3) -> list:
    """np array -> JSON-safe list, NaN -> None (so Plotly draws a gap)."""
    vals = np.asarray(arr, dtype=float).tolist()
    return [None if not np.isfinite(v) else round(v, nd) for v in vals]


def _moment_series(
    sum1: pd.Series, sum2: pd.Series, n: "int | pd.Series", x: list, nd: int = 3, clip01: bool = False
) -> dict:
    """Per-bin mean + 95% CI from per-bin sum(x) and sum(x^2). ``n`` is the per-bin
    user count (a scalar to broadcast, or a Series): all users for count/amount
    metrics (zeros included), or only users with a delta for the ratio metrics."""
    s1 = np.asarray(sum1.fillna(0.0), dtype=float)
    s2 = np.asarray(sum2.fillna(0.0), dtype=float)
    nn = np.asarray(n.fillna(0.0), dtype=float) if isinstance(n, pd.Series) else np.full(len(x), float(n))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(nn > 0, s1 / nn, np.nan)
        var = np.clip(np.where(nn > 1, (s2 - nn * mean**2) / (nn - 1), np.nan), 0, None)
        half = 1.96 * np.sqrt(var / nn)
    lo, hi = mean - half, mean + half
    if clip01:
        lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    return {"x": x, "mean": _round_list(mean, nd), "lo": _round_list(lo, nd), "hi": _round_list(hi, nd)}


def _prop_series(present, n_users: int, x: list) -> dict:
    """Per-bin proportion present/N with a 95% Wald CI, clamped to [0, 1]."""
    k = np.asarray(present.fillna(0.0), dtype=float)
    if n_users <= 0:
        nan = np.full(len(x), np.nan)
        return {"x": x, "mean": _round_list(nan, 4), "lo": _round_list(nan, 4), "hi": _round_list(nan, 4)}
    p = k / n_users
    half = 1.96 * np.sqrt(np.clip(p * (1 - p), 0, None) / n_users)
    return {
        "x": x,
        "mean": _round_list(p, 4),
        "lo": _round_list(np.clip(p - half, 0, 1), 4),
        "hi": _round_list(np.clip(p + half, 0, 1), 4),
    }


def _count_series(counts: pd.Series, x: list) -> dict:
    """Per-bin exact count (no CI: it's a population count, not a sample estimate)."""
    c = _round_list(np.asarray(counts.fillna(0.0), dtype=float), 1)
    return {"x": x, "mean": c, "lo": c, "hi": c}


def _bin_series(ub: pd.DataFrame, n_users: int, n_bins: int) -> dict:
    """Per-bin mean+CI for every bet metric, from the per-(user, bin) aggregate ``ub``."""
    ub = ub.assign(
        cnt2=ub["cnt"] ** 2,
        sbet2=ub["sbet"] ** 2,
        rup2=ub["rup"] ** 2,
        rdn2=ub["rdn"] ** 2,
        is_up=(ub["n_up"] > 0).astype(float),  # user raised their bet at least once in the bin
        is_dn=(ub["n_dn"] > 0).astype(float),
    )
    agg = cast(
        pd.DataFrame,
        ub.groupby("bin")
        .agg(
            sum_cnt=("cnt", "sum"),
            sum_cnt2=("cnt2", "sum"),
            sum_sbet=("sbet", "sum"),
            sum_sbet2=("sbet2", "sum"),
            present=("cnt", "size"),  # one row per user-in-bin, so size == users with >=1 bet
            sum_rup=("rup", "sum"),
            sum_rup2=("rup2", "sum"),
            sum_rdn=("rdn", "sum"),
            sum_rdn2=("rdn2", "sum"),
            rcnt=("rup", "count"),  # users with a within-window delta by this bin
            users_up=("is_up", "sum"),
            users_down=("is_dn", "sum"),
        )
        .reindex(range(n_bins)),
    )
    x = list(range(n_bins))

    def c(name: str) -> pd.Series:
        return cast(pd.Series, agg[name])

    return {
        # raw per-bin totals (not divided by users)
        "sum_n_bets": _count_series(c("sum_cnt"), x),
        "sum_total_bet": _count_series(c("sum_sbet"), x),
        "n_with_bet": _count_series(c("present"), x),
        # per-active-user means: divide by users who bet in the bin (present), not the cohort
        "n_bets": _moment_series(c("sum_cnt"), c("sum_cnt2"), c("present"), x),
        "total_bet": _moment_series(c("sum_sbet"), c("sum_sbet2"), c("present"), x),
        "with_bet": _prop_series(c("present"), n_users, x),
        "ratio_up": _moment_series(c("sum_rup"), c("sum_rup2"), c("rcnt"), x, nd=4, clip01=True),
        "ratio_down": _moment_series(c("sum_rdn"), c("sum_rdn2"), c("rcnt"), x, nd=4, clip01=True),
        "n_up_users": _count_series(c("users_up"), x),
        "r_up_users": _prop_series(c("users_up"), n_users, x),
        "n_down_users": _count_series(c("users_down"), x),
        "r_down_users": _prop_series(c("users_down"), n_users, x),
    }


def bet_series_config(df: pd.DataFrame, window_seconds: int, bin_seconds: int) -> dict:
    """Per-bin (mean + 95% CI) time series of each bet metric, {origin: {group: {metric: series}}}.

    Same uneven-free, fixed-width binning as the window: ``bin_seconds`` bins over
    the first ``window_seconds``. Each metric is one value per user per bin (count
    / summed bet / per-user up-down delta ratio), aggregated across the group's
    users; the bet-increase / -decrease ratio uses, per user, the share of that
    user's consecutive bet deltas falling in the bin that go up / down. A delta is
    attached to the bin of the later bullet, so users with a single bullet in the
    window contribute no delta.
    """
    bullets = df.dropna(subset=["bullet_id"]).drop_duplicates(["user_id", "bullet_id"])
    base = cast(
        pd.DataFrame,
        bullets.groupby("user_id").agg(
            account_created=("account_created", "first"),
            first_bet=("event_timestamp", "min"),
            strategy_group=("strategy_name", "first"),
        ),
    )
    strat = base["strategy_group"]
    n_bins = int(window_seconds // bin_seconds)
    out: dict = {}
    for origin_col, origin_label, _ in ORIGINS:
        origin_time = cast(pd.Series, base[origin_col]).rename("origin_time")
        m = bullets[["user_id", "event_timestamp", "bullet_id", "bet"]].merge(
            origin_time, left_on="user_id", right_index=True
        )
        off = (m["event_timestamp"] - m["origin_time"]).dt.total_seconds()
        keep = (off >= 0) & (off < window_seconds)
        m = m.loc[keep].copy()
        m["bin"] = (off.loc[keep] // bin_seconds).astype(int)
        m = m.sort_values(["user_id", "event_timestamp", "bullet_id"])
        delta = m.groupby("user_id")["bet"].diff()  # NaN only on each user's first window bullet
        m["up"] = (delta > 0).astype(float)
        m["dn"] = (delta < 0).astype(float)
        m["hasd"] = delta.notna().astype(float)  # bullet carries a valid bet-to-bet delta

        ub = (
            m.groupby(["user_id", "bin"], sort=False)
            .agg(
                cnt=("bet", "size"),
                sbet=("bet", "sum"),
                n_up=("up", "sum"),
                n_dn=("dn", "sum"),
                n_delta=("hasd", "sum"),
            )
            .reset_index()
        )
        n_delta = ub["n_delta"].where(ub["n_delta"] > 0)  # no delta in bin -> NaN ratio
        ub["rup"] = ub["n_up"] / n_delta
        ub["rdn"] = ub["n_dn"] / n_delta
        ub["strategy_group"] = ub["user_id"].map(strat)

        groups = [("All users", ub, int(len(base)))]
        for grp in sorted(strat.dropna().unique()):
            groups.append((str(grp), ub.loc[ub["strategy_group"] == grp], int((strat == grp).sum())))
        out[origin_label] = {label: _bin_series(sub, n_users, n_bins) for label, sub, n_users in groups}
    return out


def _bet_axis_ticks(bin_seconds: int, window_seconds: int) -> tuple:
    """5-minute tick marks across the bet-series figure (bin-index positions)."""
    vals, texts = [], []
    for s in range(0, window_seconds + 1, 300):
        vals.append(s // bin_seconds)
        texts.append("0" if s == 0 else f"{s // 60}m")
    return vals, texts


DAY_SECONDS = 86400


def _n_fine_bins(bin_seconds: int) -> int:
    return DAY_SECONDS // bin_seconds


def bin_event_counts(events: pd.DataFrame, origin_col: str, bin_seconds: int) -> pd.DataFrame:
    """Count users per (event, time bin) since origin_col, on a compressed ordinal axis.

    Uneven bins: bin_seconds resolution within the first 24 h, one-day bins
    beyond (accounts can predate first FM01 play by months, and a uniform
    seconds axis would squash the first day into a sliver). bin_index is the
    ordinal position; the seconds->index mapping is mirrored in the JS range
    controls (controls_js).
    """
    n_fine = _n_fine_bins(bin_seconds)
    records = []
    for event in EVENT_LABELS:
        offsets = (events[event] - events[origin_col]).dt.total_seconds().dropna().clip(lower=0)
        idx = np.where(offsets < DAY_SECONDS, offsets // bin_seconds, n_fine + offsets // DAY_SECONDS - 1)
        counts = pd.Series(idx.astype(int)).value_counts().sort_index()
        records.append(pd.DataFrame({"bin_index": counts.index, "count": counts.values, "event": event}))
    return pd.concat(records, ignore_index=True)


def _axis_ticks(bin_seconds: int, max_index: int) -> tuple:
    """Tick positions/labels: time-of-day marks across the fine region, day marks after."""
    fine_marks = [0, 300, 900, 1800, 3600, 2 * 3600, 4 * 3600, 8 * 3600, 12 * 3600, 18 * 3600]
    labels = ["0", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "18h"]
    vals = [s // bin_seconds for s in fine_marks]
    texts = list(labels)
    n_fine = _n_fine_bins(bin_seconds)
    max_day = max_index - n_fine + 2 if max_index >= n_fine else 0
    if max_day > 0:
        step = max(1, round(max_day / 8))
        for d in range(1, max_day + 1, step):
            vals.append(n_fine + d - 1)
            texts.append(f"{d}d")
    return vals, texts


def _fmt_duration(seconds: float) -> str:
    if pd.isna(seconds):
        return "n/a"
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def interpretation_html(events: pd.DataFrame, high_fish_value: float) -> str:
    """Auto-generated reading of the headline numbers; recomputed from the data on every run."""
    n = len(events)
    if n == 0:
        return "<p>No users in dataset.</p>"

    bet_delay = (events["first_bet"] - events["account_created"]).dt.total_seconds()
    kill_after_bet = (events["first_kill"] - events["first_bet"]).dt.total_seconds().dropna()
    high_after_bet = (events["first_high_kill"] - events["first_bet"]).dt.total_seconds().dropna()
    play_time = (events["stop_play"] - events["first_bet"]).dt.total_seconds()
    dep_after_acct = (events["first_deposit"] - events["account_created"]).dt.total_seconds().dropna()

    pct_kill = 100 * len(kill_after_bet) / n
    pct_high = 100 * len(high_after_bet) / n
    pct_kill_10s = 100 * (kill_after_bet <= 10).mean() if len(kill_after_bet) else 0
    pct_quit_2min = 100 * (play_time <= 120).mean()
    pct_dep = 100 * events["first_deposit"].notna().mean()
    pct_dep2 = 100 * events["second_deposit"].notna().mean()
    pct_wd = 100 * events["first_withdrawal"].notna().mean()

    has_high = events["first_high_kill"].notna()
    play_with_high = play_time[has_high].median()
    play_without_high = play_time[~has_high].median()

    bullets = [
        f"<b>Onboarding is fast:</b> half of new users fire their first bullet within "
        f"{_fmt_duration(bet_delay.median())} of account creation "
        f"(90% within {_fmt_duration(bet_delay.quantile(0.9))}).",
        f"<b>First kill comes almost immediately:</b> {pct_kill:.0f}% of users kill at least one fish in their "
        f"first session, with a median of {_fmt_duration(kill_after_bet.median())} after the first bet; "
        f"{pct_kill_10s:.0f}% of those killers get it within 10 seconds. Early kill feedback is effectively "
        f"guaranteed by game design.",
        f"<b>High-value kills (fish value &ge; {high_fish_value:g}) are a rare, late event:</b> only "
        f"{pct_high:.1f}% of users reach one, at a median of {_fmt_duration(high_after_bet.median())} into play "
        f"&mdash; long after the median player has already stopped "
        f"({_fmt_duration(play_time.median())} of play).",
        f"<b>The first session is short for most:</b> median first-session play time is "
        f"{_fmt_duration(play_time.median())}, and {pct_quit_2min:.0f}% of users stop within 2 minutes of their "
        f"first bet; the p90 session runs {_fmt_duration(play_time.quantile(0.9))}.",
        f"<b>Deposit conversion:</b> {pct_dep:.1f}% of users make at least one deposit in the observed window "
        f"(median {_fmt_duration(dep_after_acct.median())} after account creation), {pct_dep2:.1f}% make a "
        f"second, and {pct_wd:.1f}% reach a first withdrawal.",
        f"<b>High-value kills coincide with much longer sessions:</b> users who hit one play a median of "
        f"{_fmt_duration(play_with_high)} vs {_fmt_duration(play_without_high)} for everyone else. This is "
        f"correlation, not causation &mdash; longer play also gives more chances to hit a big fish.",
        "<b>Caveat:</b> only the first bet session is observed, scanned up to 12 h after first play, so "
        "&ldquo;stop play&rdquo; means the end of the first session, not churn; users may return later.",
    ]
    return "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"


def _strategy_groups(events: pd.DataFrame) -> list:
    """('All users' + each strategy level), each as (key, label, subset)."""
    groups = [("__all__", "All users", events)]
    for grp in sorted(events["strategy_group"].dropna().unique()):
        groups.append((grp, str(grp), events.loc[events["strategy_group"] == grp]))
    return groups


DEFAULT_METRICS = ("first_kill",)

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# Pure JS (no f-string) — reads the embedded CFG object and redraws one figure
# from the selected origin / groups / metrics. secToIdx / binLabel mirror the
# uneven binning in bin_event_counts so the range inputs and hover read in real
# seconds while the axis stays on ordinal bin indices.
APP_JS = """
<script>
const DASH = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'];
function pad(n) { return n < 10 ? '0' + n : '' + n; }
function fmtHms(sec) {
    var h = Math.floor(sec / 3600), r = sec % 3600, m = Math.floor(r / 60), s = r % 60;
    return h ? h + ':' + pad(m) + ':' + pad(s) : m + ':' + pad(s);
}
function binLabel(idx) {
    if (idx < CFG.nFine) {
        var st = idx * CFG.binSeconds;
        return fmtHms(st) + '\\u2013' + fmtHms(st + CFG.binSeconds);
    }
    var d = idx - CFG.nFine + 1;
    return 'day ' + d + '\\u2013' + (d + 1);
}
function secToIdx(sec) {
    if (sec < 86400) return sec / CFG.binSeconds;
    return CFG.nFine + Math.floor(sec / 86400) - 1;
}
function selOrigin() { return document.querySelector('input[name=origin]:checked').value; }
function selGroups() {
    // preserve CFG.groups order so each group's dash style is stable
    return CFG.groups.filter(function (g) {
        var cb = document.querySelector('.group[value="' + g + '"]');
        return cb && cb.checked;
    });
}
function selMetrics() {
    return CFG.order.filter(function (m) {
        var cb = document.querySelector('.metric[value="' + m + '"]');
        return cb && cb.checked;
    });
}
function isChecked(id) { var cb = document.getElementById(id); return cb && cb.checked; }
function redraw() {
    var origin = selOrigin(), groups = selGroups(), metrics = selMetrics();
    var abs = isChecked('absToggle'), useLog = isChecked('logToggle'), traces = [];
    groups.forEach(function (g) {
        var gColor = CFG.groupColors[CFG.groups.indexOf(g)];
        metrics.forEach(function (m) {
            var s = CFG.data[origin][g][m];
            if (!s) return;
            var nm = CFG.events[m] + (groups.length > 1 ? ' [' + g + ']' : '');
            // dash distinguishes metrics; with a single metric it adds no info, so keep solid
            var dash = metrics.length > 1 ? DASH[CFG.order.indexOf(m) % DASH.length] : 'solid';
            var hov = abs
                ? '%{customdata[0]}<br>%{y} users<extra>' + nm + '</extra>'
                : '%{customdata[0]}<br>%{y:.2%} of group (%{customdata[1]} users)<extra>' + nm + '</extra>';
            traces.push({
                x: s.x, y: abs ? s.c : s.y, mode: 'lines+markers', type: 'scatter', name: nm,
                line: { color: gColor, width: 1.5, dash: dash },
                marker: { color: gColor, size: 4 },
                customdata: s.x.map(function (idx, i) { return [binLabel(idx), s.c[i]]; }),
                hovertemplate: hov
            });
        });
    });
    var sub = ' (' + CFG.binSeconds + 's bins in first 24h, daily bins after)';
    // Log y on a percentage axis: Plotly keeps the fractional values and applies
    // the percent tickformat to the 10^k tick positions (0.1% / 1% / 10% / 100%).
    // Non-positive (zero) bins simply drop out, which is correct on a log scale.
    var yaxis = abs
        ? { title: 'Number of users', tickformat: '' }
        : { title: 'Share of group users', tickformat: '.1%' };
    if (useLog) { yaxis.type = 'log'; }
    var layout = {
        title: (abs ? 'Number of users by event' : 'Share of users by event') + ' \\u2014 time since ' + origin + sub,
        xaxis: {
            title: 'Time since ' + origin, tickvals: CFG.ticks.vals, ticktext: CFG.ticks.text,
            rangeslider: { visible: true, thickness: 0.08 }, range: CFG.defaultRange.slice()
        },
        yaxis: yaxis, height: 700, showlegend: true, legend: { orientation: 'h', y: -0.35 }
    };
    Plotly.react('figure', traces, layout);
    rebuildTable(origin, groups, metrics);
}
function rebuildTable(origin, groups, metrics) {
    var rows = '';
    groups.forEach(function (g) {
        metrics.forEach(function (m) {
            var st = CFG.summary[origin][g][m];
            rows += '<tr><td>' + g + '</td><td>' + CFG.events[m] + '</td><td>' + st.reached
                + '</td><td>' + st.pct + '</td><td>' + (st.median == null ? '\\u2013' : st.median)
                + '</td><td>' + (st.p90 == null ? '\\u2013' : st.p90) + '</td></tr>';
        });
    });
    document.getElementById('tbl').innerHTML = rows || '<tr><td colspan="6">No metric selected.</td></tr>';
}
function applyRange() {
    var lo = parseFloat(document.getElementById('rng-min').value);
    var hi = parseFloat(document.getElementById('rng-max').value);
    Plotly.relayout('figure', { 'xaxis.range': [secToIdx(lo), secToIdx(hi)] });
}
function resetRange() { Plotly.relayout('figure', { 'xaxis.range': CFG.defaultRange.slice() }); }

// --- First-N-minute bet behavior ---------------------------------------
// One interactive line figure: each bet metric is one value per user per
// bin_seconds bin (count, summed bet, or per-user up/down delta ratio),
// aggregated across the group's users to a per-bin mean with a 95% CI band.
// Shares the origin + group controls with the event-timing figure; capped to
// the first --window-minutes. The summary table is the 40-min-window aggregate.
function betColor(g) { return CFG.groupColors[CFG.groups.indexOf(g)]; }
function hexToRgba(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
}
function selBetMetrics() {
    return CFG.betOrder.filter(function (m) {
        var cb = document.querySelector('.betmetric[value="' + m + '"]');
        return cb && cb.checked;
    });
}
function betBinLabel(idx) {
    var st = idx * CFG.binSeconds;
    return fmtHms(st) + '\\u2013' + fmtHms(st + CFG.binSeconds);
}
// Ratio metrics live on a percentage left axis; count / per-user-mean metrics on
// a raw right axis. A second axis appears only when both kinds are on screen.
var BET_RATIO_METRICS = { with_bet: 1, ratio_up: 1, ratio_down: 1, r_up_users: 1, r_down_users: 1 };
function redrawBetSeries() {
    var origin = selOrigin(), groups = selGroups(), metrics = selBetMetrics();
    var showCI = isChecked('betCiToggle'), useLog = isChecked('betLogToggle'), traces = [];
    var ratioMetrics = metrics.filter(function (m) { return BET_RATIO_METRICS[m]; });
    var countMetrics = metrics.filter(function (m) { return !BET_RATIO_METRICS[m]; });
    var hasRatio = ratioMetrics.length > 0, hasCount = countMetrics.length > 0;
    function axisTitle(keys) { return keys.map(function (k) { return CFG.betMetrics[k]; }).join(' / '); }
    groups.forEach(function (g) {
        var gColor = betColor(g);
        metrics.forEach(function (m) {
            var s = CFG.betSeries[origin][g] && CFG.betSeries[origin][g][m];
            if (!s) return;
            var isRatio = !!BET_RATIO_METRICS[m];
            // counts sit on y2 only when ratios share the figure; otherwise everything is on y
            var yref = isRatio ? 'y' : (hasRatio ? 'y2' : 'y');
            var nm = CFG.betMetrics[m] + (groups.length > 1 ? ' [' + g + ']' : '');
            var dash = metrics.length > 1 ? DASH[CFG.betOrder.indexOf(m) % DASH.length] : 'solid';
            if (showCI) {
                // band drawn as lower bound then upper bound with fill back to it
                traces.push({
                    x: s.x, y: s.lo, mode: 'lines', line: { width: 0 }, yaxis: yref,
                    showlegend: false, hoverinfo: 'skip', connectgaps: false
                });
                traces.push({
                    x: s.x, y: s.hi, mode: 'lines', line: { width: 0 }, fill: 'tonexty', yaxis: yref,
                    fillcolor: hexToRgba(gColor, 0.15), showlegend: false, hoverinfo: 'skip', connectgaps: false
                });
            }
            var vf = isRatio ? ':.2%' : '';
            traces.push({
                x: s.x, y: s.mean, mode: 'lines+markers', type: 'scatter', name: nm, yaxis: yref,
                line: { color: gColor, width: 1.5, dash: dash }, marker: { color: gColor, size: 3 },
                connectgaps: false,
                customdata: s.x.map(function (idx, i) { return [betBinLabel(idx), s.lo[i], s.hi[i]]; }),
                hovertemplate: '%{customdata[0]}<br>%{y' + vf + '} (95% CI %{customdata[1]' + vf + '}'
                    + '\\u2013%{customdata[2]' + vf + '})<extra>' + nm + '</extra>'
            });
        });
    });
    var ratioAxis = { title: axisTitle(ratioMetrics), tickformat: '.0%', rangemode: 'tozero' };
    var countAxis = { title: axisTitle(countMetrics), rangemode: 'tozero' };
    var layout = {
        title: 'Bet behavior over time \\u2014 time since ' + origin
            + ' (' + CFG.binSeconds + 's bins, first ' + CFG.windowMinutes + ' min)',
        xaxis: {
            title: 'Time since ' + origin, tickvals: CFG.betTicks.vals, ticktext: CFG.betTicks.text,
            rangeslider: { visible: true, thickness: 0.08 }, range: [0, CFG.nBetBins]
        },
        height: 560, showlegend: true, legend: { orientation: 'h', y: -0.3 }
    };
    if (hasRatio && hasCount) {
        layout.yaxis = ratioAxis;
        countAxis.overlaying = 'y'; countAxis.side = 'right';
        layout.yaxis2 = countAxis;
    } else if (hasRatio) {
        layout.yaxis = ratioAxis;
    } else {
        layout.yaxis = countAxis;
    }
    // log y keeps the percent tickformat; Plotly applies it to the 10^k positions
    if (useLog) { layout.yaxis.type = 'log'; if (layout.yaxis2) { layout.yaxis2.type = 'log'; } }
    Plotly.react('betFig', traces, layout);
}
function fmtCi(st, pct) {
    if (!st || st.mean == null) return '\\u2013';
    var f = pct ? function (x) { return (100 * x).toFixed(1) + '%'; } : function (x) { return (+x).toFixed(2); };
    return f(st.mean) + ' [' + f(st.lo) + ', ' + f(st.hi) + ']';
}
function usersCell(st) {
    if (!st || st.mean == null) return '\\u2013';
    return st.reached + ' (' + (100 * st.mean).toFixed(1) + '%)';
}
function betTable() {
    var origin = selOrigin(), groups = selGroups(), bets = CFG.bets[origin], rows = '';
    groups.forEach(function (g) {
        var b = bets[g];
        if (!b) return;
        rows += '<tr><td>' + g + '</td><td>' + fmtCi(b.n_bets, false) + '</td><td>' + fmtCi(b.total_bet, false)
            + '</td><td>' + fmtCi(b.with_bet, true) + '</td><td>' + fmtCi(b.ratio_up, true)
            + '</td><td>' + fmtCi(b.ratio_down, true) + '</td><td>' + b.with_bet.reached + ' / ' + b.n_users
            + '</td><td>' + usersCell(b.users_up) + '</td><td>' + usersCell(b.users_down) + '</td></tr>';
    });
    document.getElementById('betTbl').innerHTML = rows || '<tr><td colspan="9">No group selected.</td></tr>';
}
function applyRangeBet() {
    var lo = parseFloat(document.getElementById('rng-min-bet').value);
    var hi = parseFloat(document.getElementById('rng-max-bet').value);
    Plotly.relayout('betFig', { 'xaxis.range': [lo / CFG.binSeconds, hi / CFG.binSeconds] });
}
function resetRangeBet() { Plotly.relayout('betFig', { 'xaxis.range': [0, CFG.nBetBins] }); }

document.querySelectorAll('input[name=origin], .group, .metric, #absToggle, #logToggle').forEach(function (el) {
    el.addEventListener('change', redraw);
});
document.querySelectorAll('input[name=origin], .group').forEach(function (el) {
    el.addEventListener('change', function () { betTable(); redrawBetSeries(); });
});
document.querySelectorAll('.betmetric, #betCiToggle, #betLogToggle').forEach(function (el) {
    el.addEventListener('change', redrawBetSeries);
});
redraw();
betTable();
redrawBetSeries();
</script>
"""


def _report_config(events: pd.DataFrame, df: pd.DataFrame, bin_seconds: int, window_seconds: int) -> dict:
    """Precompute every (origin, group, metric) series + summary stat for the client."""
    groups = _strategy_groups(events)
    data: dict = {}
    summary: dict = {}
    max_idx = _n_fine_bins(bin_seconds)
    for origin_col, origin_label, _ in ORIGINS:
        data[origin_label] = {}
        summary[origin_label] = {}
        for _, label, sub in groups:
            binned = bin_event_counts(sub, origin_col, bin_seconds)
            n_users = len(sub)
            dser: dict = {}
            sser: dict = {}
            for event in EVENT_LABELS:
                d = binned[binned["event"] == event]
                xs = [int(v) for v in d["bin_index"].tolist()]
                cs = [int(v) for v in d["count"].tolist()]
                # y is the share of the group's users per bin so groups of
                # different sizes are comparable; raw counts kept in c for hover.
                ys = [round(c / n_users, 6) if n_users else 0 for c in cs]
                if xs:
                    max_idx = max(max_idx, max(xs))
                dser[event] = {"x": xs, "y": ys, "c": cs}
                offsets = (sub[event] - sub[origin_col]).dt.total_seconds().dropna()
                sser[event] = {
                    "reached": int(len(offsets)),
                    "pct": round(100 * len(offsets) / n_users, 1) if n_users else 0,
                    "median": round(float(offsets.median()), 1) if len(offsets) else None,
                    "p90": round(float(offsets.quantile(0.9)), 1) if len(offsets) else None,
                }
            data[origin_label][label] = dser
            summary[origin_label][label] = sser

    tickvals, ticktext = _axis_ticks(bin_seconds, max_idx)
    group_labels = [label for _, label, _ in groups]
    return {
        "events": dict(EVENT_LABELS),
        "order": list(EVENT_LABELS),
        "groups": group_labels,
        "groupColors": [_group_color(i) for i in range(len(group_labels))],
        "ticks": {"vals": [int(v) for v in tickvals], "text": ticktext},
        "binSeconds": bin_seconds,
        "nFine": _n_fine_bins(bin_seconds),
        "defaultRange": [0, window_seconds // bin_seconds],
        "windowSeconds": window_seconds,
        "windowMinutes": round(window_seconds / 60, 1),
        "betMetrics": dict(BET_METRICS),
        "betOrder": list(BET_METRICS),
        "bets": bet_window_config(df, window_seconds),
        "betSeries": bet_series_config(df, window_seconds, bin_seconds),
        "betTicks": {
            "vals": [int(v) for v in _bet_axis_ticks(bin_seconds, window_seconds)[0]],
            "text": _bet_axis_ticks(bin_seconds, window_seconds)[1],
        },
        "nBetBins": int(window_seconds // bin_seconds),
        "data": data,
        "summary": summary,
    }


def _controls_html(group_labels: list, bin_seconds: int, window_seconds: int) -> str:
    origin_radios = "".join(
        f'<label><input type="radio" name="origin" value="{ol}" {"checked" if i == 0 else ""}> {ol}</label>'
        for i, (_, ol, _) in enumerate(ORIGINS)
    )
    group_boxes = "".join(
        f'<label style="color:{_group_color(i)}">'
        f'<input type="checkbox" class="group" value="{g}" {"checked" if g == "All users" else ""}> {g}</label>'
        for i, g in enumerate(group_labels)
    )
    metric_boxes = "".join(
        f'<label><input type="checkbox" class="metric" value="{e}" {"checked" if e in DEFAULT_METRICS else ""}>'
        f" {lab}</label>"
        for e, lab in EVENT_LABELS.items()
    )
    return (
        f'<div class="ctl-row"><b>Origin:</b> {origin_radios}'
        f' &nbsp;&nbsp; <b>Y-axis:</b> <label><input type="checkbox" id="absToggle">'
        f" show absolute user count (default: ratio)</label>"
        f' &nbsp; <label><input type="checkbox" id="logToggle"> log scale</label></div>'
        f'<div class="ctl-row groups"><b>Groups (one color each):</b> {group_boxes}</div>'
        f'<div class="ctl-row metrics">{metric_boxes}</div>'
        f'<div class="ctl-row"><b>Time range (s):</b> '
        f'<input type="number" id="rng-min" value="0" step="{bin_seconds}" style="width:90px"> &ndash; '
        f'<input type="number" id="rng-max" value="{window_seconds}" step="{bin_seconds}" style="width:90px"> '
        f'<button onclick="applyRange()">Apply</button> '
        f'<button onclick="resetRange()">Reset</button></div>'
    )


def _bet_controls_html(bin_seconds: int, window_seconds: int) -> str:
    # one labelled row per metric group so the now-dozen metrics stay readable
    group_rows = "".join(
        '<div class="ctl-row metrics"><span class="grp-label">'
        + header
        + ":</span> "
        + "".join(
            f'<label><input type="checkbox" class="betmetric" value="{k}"'
            f' {"checked" if k in DEFAULT_BET_METRICS else ""}> {BET_METRICS[k]}</label>'
            for k in keys
        )
        + "</div>"
        for header, keys in BET_METRIC_GROUPS
    )
    return (
        f"{group_rows}"
        f'<div class="ctl-row"><label><input type="checkbox" id="betCiToggle" checked> show 95% CI band</label>'
        f' &nbsp; <label><input type="checkbox" id="betLogToggle"> log scale</label></div>'
        f'<div class="ctl-row"><b>Time range (s):</b> '
        f'<input type="number" id="rng-min-bet" value="0" step="{bin_seconds}" style="width:90px"> &ndash; '
        f'<input type="number" id="rng-max-bet" value="{window_seconds}" step="{bin_seconds}" style="width:90px"> '
        f'<button onclick="applyRangeBet()">Apply</button> '
        f'<button onclick="resetRangeBet()">Reset</button></div>'
    )


def _bet_behavior_figure(cfg: dict, origin_label: str = "first bet"):
    """Render and return the 2x2 bet-behavior matplotlib Figure (All users)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bs = cfg["betSeries"][origin_label]["All users"]
    bin_s = cfg["binSeconds"]
    win_min = cfg["windowMinutes"]
    nb = cfg["nBetBins"]
    n_users = cfg["bets"][origin_label]["All users"]["n_users"]
    x = np.array([i * bin_s / 60.0 for i in range(nb)])  # minutes since origin

    def arr(metric: str, key: str) -> np.ndarray:
        return np.array([np.nan if v is None else v for v in bs[metric][key]], dtype=float)

    def panel(ax, metrics: list, title: str, ylabel: str, pct: bool = False) -> None:
        for metric, color in zip(metrics, ["#377eb8", "#e41a1c"]):
            mean, lo, hi = arr(metric, "mean"), arr(metric, "lo"), arr(metric, "hi")
            if pct:
                mean, lo, hi = mean * 100, lo * 100, hi * 100
            ax.plot(x, mean, color=color, lw=1.6, label=cfg["betMetrics"][metric])
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"minutes since {origin_label}")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, win_min)
        ax.grid(True, alpha=0.3)
        if len(metrics) > 1:
            ax.legend(fontsize=8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panel(axes[0, 0], ["with_bet"], "Active users (share of cohort with a bet)", "share (%)", pct=True)
    panel(axes[0, 1], ["n_bets"], "Avg bets per active user", "bets / active user")
    panel(axes[1, 0], ["total_bet"], "Avg bet amount per active user", "amount / active user")
    panel(axes[1, 1], ["r_up_users", "r_down_users"], "Users adjusting their bet", "share (%)", pct=True)
    fig.suptitle(
        f"FM01 FTUE \u2014 first-{win_min:g}-min bet behavior (all users, {bin_s}s bins, N={n_users:,})", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


ORIGIN_ZH = {"account created": "账号注册", "first bet": "首次下注"}


def _bet_md_interpretation(cfg: dict, origin_label: str = "first bet") -> list:
    """Auto-generated reading of one origin's bet-behavior headline numbers (recomputed each run)."""
    bs = cfg["betSeries"][origin_label]["All users"]
    bets = cfg["bets"][origin_label]["All users"]
    bin_s = cfg["binSeconds"]
    win_str = f"{cfg['windowMinutes']:g}"
    means = bs["with_bet"]["mean"]

    def t_of(i: int) -> float:
        return i * bin_s / 60.0

    peak_i = max(range(len(means)), key=lambda i: means[i] if means[i] is not None else -1.0)
    peak_v = means[peak_i] or 0.0
    bin0 = means[0] or 0.0

    def below_after(thr: float) -> str:
        for i in range(peak_i, len(means)):
            v = means[i]
            if v is not None and v < thr:
                return f"{t_of(i):.0f} 分钟"
        return "—"

    def last_val(key: str) -> float:
        for v in reversed(bs[key]["mean"]):
            if v is not None:
                return v
        return 0.0

    nb0 = bs["n_bets"]["mean"][0] or 0.0
    nb_last = last_val("n_bets")
    zh = ORIGIN_ZH.get(origin_label, origin_label)
    return [
        f"**活跃节奏**：以「{zh}」为起点，首个 {bin_s} 秒箱有 {100 * bin0:.0f}% 的用户在下注；"
        f"活跃占比在第 {t_of(peak_i):.1f} 分钟达到峰值 {100 * peak_v:.0f}%，约 {below_after(0.5)}后降到 50% 以下，"
        f"{below_after(0.1)}后降到 10% 以下（见左上）。",
        f"**人均强度不降反升**：按活跃用户计的人均下注次数从约 {nb0:.0f} 升到后段约 {nb_last:.0f}（右上），"
        "人均下注金额同样走高（左下）；留下来的是重度玩家——休闲玩家流失后，仍在场的用户下注更密集。",
        f"**极少人调整注额**：{win_str} 分钟内曾提高 / 降低注额的用户仅约 "
        f"{100 * bets['users_up']['mean']:.0f}% / {100 * bets['users_down']['mean']:.0f}%，"
        "逐弹注额几乎不变；多数玩家固定注额自动连发（右下）。",
    ]


def _bet_origin_section_md(cfg: dict, origin_label: str, num: str, img_src: str) -> str:
    """One per-origin markdown section: figure + interpretation + 40-min summary table."""
    win_str = f"{cfg['windowMinutes']:g}"
    bets = cfg["bets"][origin_label]["All users"]
    n_users = bets["n_users"]
    zh = ORIGIN_ZH.get(origin_label, origin_label)
    interp = "\n".join(f"- {b}" for b in _bet_md_interpretation(cfg, origin_label))
    return f"""## {num}、以「{zh}」为起点（origin = {origin_label}）

![{zh}起点 · 首 {win_str} 分钟下注行为]({img_src})

{interp}

**{win_str} 分钟窗口汇总（全体用户）**
| 指标 | 值（95% CI） |
|---|---|
| 人均下注次数（活跃用户）Avg bets / active user | {bets['n_bets']['mean']:.0f} [{bets['n_bets']['lo']:.0f}, {bets['n_bets']['hi']:.0f}] |
| 人均下注金额（活跃用户）Avg bet amount / active user | {bets['total_bet']['mean']:.0f} [{bets['total_bet']['lo']:.0f}, {bets['total_bet']['hi']:.0f}] |
| 窗口内有下注的用户 Users with a bet | {bets['with_bet']['reached']:,} / {n_users:,}（{100 * bets['with_bet']['mean']:.1f}%） |
| 曾提高注额的用户 Users increasing bet | {bets['users_up']['reached']:,}（{100 * bets['users_up']['mean']:.1f}%） |
| 曾降低注额的用户 Users decreasing bet | {bets['users_down']['reached']:,}（{100 * bets['users_down']['mean']:.1f}%） |
"""


def build_bet_behavior_markdown(
    cfg: dict, output_path: str, origins: tuple = ("account created", "first bet"), embed: bool = True
) -> None:
    """Write a markdown summary of the first-window bet behavior, one section per origin.

    Sections are emitted in ``origins`` order (account-register first, then first-bet).
    ``embed=True`` inlines each 2x2 figure as a base64 data URI (single portable file;
    renders in VS Code / most viewers but NOT on GitHub). ``embed=False`` writes one
    sidecar PNG per origin next to the markdown and links it relatively (GitHub-renderable).
    Interpretation and tables are recomputed from ``cfg``.
    """
    import base64
    import io

    import matplotlib.pyplot as plt

    win_str = f"{cfg['windowMinutes']:g}"
    bin_s = cfg["binSeconds"]
    n_users = cfg["bets"][origins[0]]["All users"]["n_users"]

    def figure_src(origin_label: str) -> str:
        fig = _bet_behavior_figure(cfg, origin_label)
        if embed:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130)
            src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        else:
            slug = origin_label.replace(" ", "_")
            png_path = os.path.splitext(output_path)[0] + f"_{slug}.png"
            os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
            fig.savefig(png_path, dpi=130)
            src = os.path.basename(png_path)
        plt.close(fig)
        return src

    nums = ["三", "四", "五", "六"]
    sections = "\n".join(
        _bet_origin_section_md(cfg, origin_label, nums[i], figure_src(origin_label))
        for i, origin_label in enumerate(origins)
    )

    header = f"""# 捕鱼（FM01）新用户首 {win_str} 分钟下注行为分析（First-{win_str}-minute bet behavior）

> 配套报告：交互式 HTML（`ftue_analysis_report.html`）「First-{win_str}-minute bet behavior」一节。
> 本文为该节的静态摘要，数据口径与 FTUE 报告一致（FM01 / CNY 新用户，首次会话），由 analysis_ftue_events.py --markdown 自动生成。

## 一、口径与时间窗
- **时间起点**：分别以 **账号注册**（account created）与 **首次下注**（first bet）为 0 点，各看其后 0–{win_str} 分钟，按 **{bin_s} 秒** 分箱。
- **去重**：按 `(user_id, bullet_id)` 去重后再统计，避免交易记录扇出导致下注次数虚高。
- **样本**：全体用户 N = {n_users:,}。

## 二、「活跃用户」如何定义
对每个 {bin_s} 秒时间箱：
1. 取该用户落在该时间箱内的（去重后）子弹；
2. 按 `(user_id, bin)` 聚合，**每个用户在每个时间箱至多一行**；
3. 该时间箱的「活跃用户数」= 时间箱内有 ≥1 颗子弹的用户数（即聚合后的行数）。

因此 **活跃用户占比** = 活跃用户数 / 全体用户数 N；**人均指标（/ active user）** 的分母是**该时间箱的活跃用户数**，
而非 {win_str} 分钟内的全体用户——衡量「当下仍在玩的人」的强度，不被已离开的用户稀释。

> 说明：以**账号注册**为起点时，0 点是账号创建时刻；由于不少账号在首次玩捕鱼前已存在较久，注册后头几个箱并非 100% 活跃，
> 该视角衡量「注册→开始玩」的速度与节奏。以**首次下注**为起点时，0 点是每位用户的首颗子弹，故首个箱必为 100% 活跃，
> 衡量开玩后的下注强度与留存。
"""

    md = header + "\n" + sections + "\n> 注：图与表均可在交互报告中切换时间起点与策略分组；本文取「全体用户」。\n"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md)
    print(f"Bet-behavior markdown written to {output_path}")


def build_report(
    events: pd.DataFrame,
    cfg: dict,
    bin_seconds: int,
    window_seconds: int,
    high_fish_value: float,
    output_path: str,
) -> None:
    group_labels = [label for _, label, _ in _strategy_groups(events)]
    window_min = cfg["windowMinutes"]
    window_min_str = f"{window_min:g}"

    table_head = (
        "<table><thead><tr><th>group</th><th>event</th><th>users reached</th>"
        "<th>% of users</th><th>median (s)</th><th>p90 (s)</th></tr></thead><tbody id='tbl'></tbody></table>"
    )

    bet_table_head = (
        "<table><thead><tr><th>group</th><th>avg bets / active user</th><th>avg bet amt / active user</th>"
        "<th>% with a bet</th><th>bet-increase ratio</th><th>bet-decrease ratio</th>"
        "<th>users w/ bet / N</th><th>users increasing (count, %)</th>"
        "<th>users decreasing (count, %)</th></tr></thead><tbody id='betTbl'></tbody></table>"
    )
    bet_section = [
        f"<h2>First-{window_min_str}-minute bet behavior</h2>",
        f"<p>Per-bin metrics over the first {window_min_str} minutes (measured from the selected origin), "
        f"binned at {bin_seconds}s. <b>Total bets / Total bet amount</b> are the raw per-bin sums. "
        "<b>Avg ... / active user</b> divide that sum by the users who actually bet in the bin (not the whole "
        "cohort), shown as the cross-user mean with a 95% CI band. <b>Users with a bet</b> is the count and "
        "share of the cohort active in the bin. <b>Bet-increase / -decrease ratio</b> is, per user, the share "
        "of that bin's consecutive bet-to-bet changes that go up / down (bins with no change are skipped). "
        "<b>Users increasing / decreasing bet</b> are the count and share of users with at least one increase "
        "/ decrease (any delta &gt; 0 / &lt; 0) in the bin. Bullets are de-duplicated before counting. Uses "
        "the same origin / group selectors above; ratio metrics use a percentage left axis and count / "
        "per-user-mean metrics a raw right axis (shown only when both kinds are selected), and the log toggle "
        "helps when overlaying metrics of different magnitude.</p>",
        _bet_controls_html(bin_seconds, window_seconds),
        '<div id="betFig" style="height:580px"></div>',
        f"<h3>{window_min_str}-minute window summary</h3>",
        bet_table_head,
    ]

    parts = [
        "<html><head><title>FTUE Event Timing Report</title>",
        "<style>body{font-family:sans-serif;margin:24px} table{border-collapse:collapse}",
        "td,th{padding:4px 12px;border-bottom:1px solid #ddd;text-align:right}",
        "th{background:#f5f5f5} td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}",
        ".ctl-row{margin:10px 0} .ctl-row.metrics label{margin-right:14px}",
        ".grp-label{font-weight:bold;display:inline-block;min-width:150px}",
        "select{font-size:14px} h2{margin-top:40px}</style></head><body>",
        "<h1>FTUE Event Timing Report (FM01 first session)</h1>",
        f"<p>Generated {datetime.now():%Y-%m-%d %H:%M} &middot; {len(events)} users &middot; "
        f"bin = {bin_seconds}s &middot; window = {window_min_str} min &middot; "
        f"high-value fish threshold = {high_fish_value}</p>",
        "<p>Users grouped by the strategy_name of their first bullet (session-start strategy).</p>",
        "<h2>Interpretation (All users)</h2>",
        interpretation_html(events, high_fish_value),
        "<h2>Interactive event timing</h2>",
        _controls_html(group_labels, bin_seconds, window_seconds),
        '<div id="figure" style="height:720px"></div>',
        "<h3>Selected metrics &mdash; summary</h3>",
        table_head,
        *bet_section,
        f'<script src="{PLOTLY_CDN}"></script>',
        "<script>const CFG = " + json.dumps(cfg) + ";</script>",
        APP_JS,
        "</body></html>",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(parts))
    print(f"Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FTUE event-timing analysis report")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Parquet dataset path (S3 or local)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output HTML report path")
    parser.add_argument("--bin-seconds", type=int, default=30, help="Time bin width in seconds")
    parser.add_argument(
        "--window-minutes",
        type=float,
        default=DEFAULT_WINDOW_MINUTES,
        help="Window (minutes from origin) for the bet-behavior metrics and the default curve zoom",
    )
    parser.add_argument("--high-fish-value", type=float, default=500, help="fish_value threshold for high-value kill")
    parser.add_argument(
        "--markdown",
        nargs="?",
        const=DEFAULT_MD_OUTPUT,
        default=None,
        help=f"Also write a markdown bet-behavior summary (default path: {DEFAULT_MD_OUTPUT})",
    )
    parser.add_argument(
        "--markdown-mode",
        choices=["embed", "sidecar"],
        default="embed",
        help="embed: inline figure as base64 (one file, not GitHub-renderable); sidecar: write a PNG next to the md (GitHub-renderable)",
    )
    args = parser.parse_args()

    window_seconds = int(round(args.window_minutes * 60))
    df = load_data(args.input)
    events = compute_user_events(df, args.high_fish_value)
    cfg = _report_config(events, df, args.bin_seconds, window_seconds)
    build_report(events, cfg, args.bin_seconds, window_seconds, args.high_fish_value, args.output)
    if args.markdown:
        build_bet_behavior_markdown(cfg, args.markdown, embed=args.markdown_mode == "embed")


if __name__ == "__main__":
    setup_logging(f"{LOCAL_ROOT}/jobs/log", log_filename=os.path.splitext(os.path.basename(__file__))[0] + ".log")
    main()
