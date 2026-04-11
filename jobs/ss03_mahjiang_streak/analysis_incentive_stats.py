"""SS03 mahjong streak incentive stats: merge user-day stats with AB group file, aggregate, plot RTP, export CSV.

Summary DataFrame columns (``analyze()[0]``, also ``incentive_stats_summary.csv``)
--------------------------------------------------------------------------------
date
    Day from incentive stats, matched to ``activity_date`` in the group file.
partition_ab
    AB cohort id (UUID string); arm is inferred from the first two characters
    (``4a`` vs ``4f``) for thresholds.
num_users
    Count of distinct ``user_id`` with merged stats for this date and partition.
num_users_with_fg: users with at least one free game (not incentivized) row that day
    Users with at least one detail row where ``type == 1`` (free game) that day
    in this partition.
num_users_low_spin
    Users whose **base-game** spin total (``type == 0`` ``spin_count`` summed)
    is **strictly less** than the arm spin threshold (100 for ``4a*``, 130 for ``4f*``).
num_users_high_rtp
    Users with base-game spin total **greater than or equal** to the arm spin
    threshold **and** mean base-game ``rtp`` **greater than** the arm RTP cutoff
    (0.5 for ``4a*``, 0.6 for ``4f*``).
num_users_incentivized
    Users with any detail row where ``incentivized == 1`` that day.
num_users_high_rtp_incentivized
    Users who satisfy ``high_rtp`` **and** have ``has_incentivized`` true (any
    incentivized row that day).
median_spin_below_thresh
    Median of base-game spin totals among users in ``num_users_low_spin`` only;
    missing (NaN) when there are no users below the spin threshold for that row.
mean_rtp_no_fg_not_incentivized
    Mean of per-user base-game ``rtp`` among users with **no** ``type == 1`` row and
    **no** ``incentivized == 1`` row that day. Rows with missing ``rtp`` are ignored in
    the mean; NaN if no such user has a defined ``rtp``.
avg_base_spin_no_fg_not_incentivized
    Mean of ``total_spin`` (base-game spin sum) over the same user subset as
    ``mean_rtp_no_fg_not_incentivized``. NaN when that subset is empty.
ratio_users_with_fg
    ``num_users_with_fg / num_users``; NaN if ``num_users`` is 0.
ratio_users_low_spin
    ``num_users_low_spin / num_users``; NaN if ``num_users`` is 0.
ratio_users_high_rtp
    ``num_users_high_rtp / num_users``; NaN if ``num_users`` is 0.
ratio_users_incentivized
    ``num_users_incentivized / num_users``; NaN if ``num_users`` is 0.
ratio_users_high_rtp_incentivized
    ``num_users_high_rtp_incentivized / num_users``; NaN if ``num_users`` is 0.

Per-user working frame ``user_day`` (``analyze()[1]``)
------------------------------------------------------
total_spin
    Sum of ``spin_count`` over rows with ``type == 0`` only (base game).
has_fg
    True if any row that day has ``type == 1``.
has_incentivized
    True if any row that day has ``incentivized == 1``.
spin_thresh / rtp_thresh
    Arm-specific cutoffs copied from merged detail rows.
rtp
    Mean of ``rtp`` over ``type == 0`` rows only (missing if no base-game rows).
low_spin
    ``total_spin < spin_thresh``.
high_rtp
    ``total_spin >= spin_thresh`` and ``rtp > rtp_thresh`` (requires finite ``rtp``).
high_rtp_incentivized
    ``high_rtp`` and ``has_incentivized`` (high RTP bucket and got incentivized).
no_fg_not_incentivized
    True when the user had no free-game (``type == 1``) and no incentivized row that day.
"""

import argparse
import logging
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

from bituslabs_ds.config import LOCAL_ROOT, S3_BUCKET
from bituslabs_ds.s3_utils import upload_file_to_s3

file1 = "/Users/niuxin/Documents/data_ss03/ss03_incentive_stats.csv"
file2 = "/Users/niuxin/Documents/data_ss03/ss03_incentive_user_group_by_day.csv"

OUTPUT_DIR = Path(LOCAL_ROOT) / "jobs" / "output" / "ss03_mahjiang_streak"
S3_ANALYSIS_PREFIX = "ds-analysis/ss03_mahjiang_streak"

DEFAULT_LOCAL_FIGURE = OUTPUT_DIR / "base_game_rtp_by_day.png"
DEFAULT_S3_KEY = f"{S3_ANALYSIS_PREFIX}/base_game_rtp_by_day.png"

DEFAULT_LOCAL_SUMMARY_CSV = OUTPUT_DIR / "incentive_stats_summary.csv"
DEFAULT_S3_SUMMARY_KEY = f"{S3_ANALYSIS_PREFIX}/incentive_stats_summary.csv"


def warn_user_ids_in_multiple_partitions(groups: pd.DataFrame) -> None:
    """Log a warning when the same user_id is listed under more than one partition_ab."""
    n_part = cast(
        pd.Series,
        groups.groupby("user_id")["partition_ab"].nunique(),
    )
    multi = cast(pd.Series, n_part[n_part > 1])
    if multi.size == 0:
        return
    uids = sorted(multi.index.tolist())
    logger.warning(
        "%s user_id(s) appear in more than one partition_ab group; "
        "merge will duplicate stats rows per user. Details:",
        len(uids),
    )
    for uid in uids:
        parts = sorted(groups.loc[groups["user_id"] == uid, "partition_ab"].drop_duplicates().tolist())
        ab_arm = [str(p)[:2].lower() for p in parts]
        logger.warning(
            "  user_id=%s ab_arm=%s partition_ab=%s",
            uid,
            ab_arm,
            parts,
        )


def analyze(
    path_stats: str = file1,
    path_groups: str = file2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs, merge, aggregate. See module docstring for column definitions."""
    data1 = pd.read_csv(path_stats)
    data2 = pd.read_csv(path_groups)

    data2 = data2.copy()
    data2["partition_ab"] = data2["partition_ab"].astype(str).str.strip().str.strip('"')
    warn_user_ids_in_multiple_partitions(data2)

    m = pd.merge(data1, data2, left_on=["user_id", "date"], right_on=["user_id", "activity_date"], how="inner")
    prefix = m["partition_ab"].str[:2].str.lower()
    m["spin_thresh"] = np.select(
        [prefix == "4f", prefix == "4a"],
        [130, 100],
        default=np.nan,
    )
    m["rtp_thresh"] = np.select(
        [prefix == "4f", prefix == "4a"],
        [0.6, 0.5],
        default=np.nan,
    )
    m = m[m["spin_thresh"].notna()].copy()
    # Spin thresholds compare only base-game (type == 0) spins, not free-game rows.
    m["spin_count_base"] = np.where(m["type"] == 0, m["spin_count"], 0)

    gcols = ["date", "partition_ab", "user_id"]
    user_day = cast(
        pd.DataFrame,
        m.groupby(gcols, as_index=False).agg(
            total_spin=("spin_count_base", "sum"),
            has_fg=("type", lambda s: (s == 1).any()),
            has_incentivized=("incentivized", lambda s: (s == 1).any()),
            spin_thresh=("spin_thresh", "first"),
            rtp_thresh=("rtp_thresh", "first"),
        ),
    )
    rtp_base = cast(
        pd.DataFrame,
        m.loc[m["type"] == 0].groupby(gcols, as_index=False).agg(rtp=("rtp", "mean")),
    )
    user_day = cast(pd.DataFrame, user_day.merge(rtp_base, on=gcols, how="left"))

    user_day["low_spin"] = user_day["total_spin"] < user_day["spin_thresh"]
    user_day["high_rtp"] = (user_day["total_spin"] >= user_day["spin_thresh"]) & (
        user_day["rtp"] > user_day["rtp_thresh"]
    )
    user_day["high_rtp_incentivized"] = user_day["high_rtp"] & user_day["has_incentivized"]
    user_day["no_fg_not_incentivized"] = (~user_day["has_fg"]) & (~user_day["has_incentivized"])

    summary = user_day.groupby(["date", "partition_ab"], as_index=False).agg(
        num_users=("user_id", "count"),
        num_users_with_fg=("has_fg", "sum"),
        num_users_low_spin=("low_spin", "sum"),
        num_users_high_rtp=("high_rtp", "sum"),
        num_users_incentivized=("has_incentivized", "sum"),
        num_users_high_rtp_incentivized=("high_rtp_incentivized", "sum"),
    )
    # Median base-game spin (total_spin) among users below this row's group threshold only.
    med_below = cast(
        pd.DataFrame,
        user_day[user_day["low_spin"]]
        .groupby(["date", "partition_ab"], as_index=False)
        .agg(
            median_spin_below_thresh=("total_spin", "median"),
        ),
    )
    summary = cast(pd.DataFrame, summary.merge(med_below, on=["date", "partition_ab"], how="left"))

    pure = user_day[user_day["no_fg_not_incentivized"]]
    pure_stats = cast(
        pd.DataFrame,
        pure.groupby(["date", "partition_ab"], as_index=False).agg(
            mean_rtp_no_fg_not_incentivized=("rtp", "mean"),
            avg_base_spin_no_fg_not_incentivized=("total_spin", "mean"),
        ),
    )
    summary = cast(pd.DataFrame, summary.merge(pure_stats, on=["date", "partition_ab"], how="left"))

    summary = summary.sort_values(["date", "partition_ab"]).reset_index(drop=True)  # pyright: ignore[reportCallIssue]

    nu = summary["num_users"].replace(0, np.nan)
    summary["ratio_users_with_fg"] = summary["num_users_with_fg"] / nu
    summary["ratio_users_low_spin"] = summary["num_users_low_spin"] / nu
    summary["ratio_users_high_rtp"] = summary["num_users_high_rtp"] / nu
    summary["ratio_users_incentivized"] = summary["num_users_incentivized"] / nu
    summary["ratio_users_high_rtp_incentivized"] = summary["num_users_high_rtp_incentivized"] / nu

    return summary, user_day


def rtp_per_user_per_day(user_day: pd.DataFrame) -> pd.DataFrame:
    """One base-game RTP per user per day and AB prefix group (4a / 4f)."""
    base = user_day.dropna(subset=["rtp"]).copy()
    base["ab_group"] = base["partition_ab"].str[:2].str.lower()
    return cast(
        pd.DataFrame,
        base.groupby(["date", "ab_group", "user_id"], as_index=False).agg(
            rtp=("rtp", "mean"),
        ),
    )


def save_rtp_histplot_by_day(plot_df: pd.DataFrame, out_path: Path) -> None:
    if plot_df.empty:
        return
    df = plot_df.copy()
    df["date"] = df["date"].astype(str)
    ab_cats = sorted(df["ab_group"].astype(str).str.lower().unique())
    df["ab_group"] = pd.Categorical(
        df["ab_group"].astype(str).str.lower(),
        categories=ab_cats,
        ordered=True,
    )
    g = sns.displot(
        df,
        x="rtp",
        row="ab_group",
        col="date",
        kind="hist",
        bins=30,
        height=2.8,
        aspect=1.0,
        facet_kws={"margin_titles": True, "sharex": True, "sharey": False},
    )
    g.set_axis_labels("Base-game RTP (mean)", "Users")
    g.figure.suptitle(
        "Per-user base-game RTP (type = 0) by day and group (4a vs 4f)",
        y=1.02,
    )
    g.figure.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")


def save_summary_local_and_s3(
    summary: pd.DataFrame,
    local_path: Path = DEFAULT_LOCAL_SUMMARY_CSV,
    s3_bucket: str = S3_BUCKET,
    s3_key: str = DEFAULT_S3_SUMMARY_KEY,
) -> tuple[Path, str | None]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(local_path, index=False)
    uri = upload_file_to_s3(str(local_path), s3_bucket, s3_key)
    return local_path, uri


def save_figure_local_and_s3(
    plot_df: pd.DataFrame,
    local_path: Path = DEFAULT_LOCAL_FIGURE,
    s3_bucket: str = S3_BUCKET,
    s3_key: str = DEFAULT_S3_KEY,
) -> tuple[Path, str | None]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    save_rtp_histplot_by_day(plot_df, local_path)
    uri = upload_file_to_s3(str(local_path), s3_bucket, s3_key)
    return local_path, uri


def main() -> pd.DataFrame:
    summary, _user_day = analyze()
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="SS03 incentive stats summary and RTP histogram.")
    parser.add_argument(
        "--stats-csv",
        default=file1,
        help="Path to incentive stats CSV.",
    )
    parser.add_argument(
        "--groups-csv",
        default=file2,
        help="Path to user group CSV.",
    )
    args = parser.parse_args()

    summary, user_day = analyze(args.stats_csv, args.groups_csv)
    print(summary.to_string(index=False))

    csv_path, csv_uri = save_summary_local_and_s3(summary)
    print(f"Wrote summary CSV to {csv_path.resolve()}")
    if csv_uri:
        print(f"Uploaded summary CSV to {csv_uri}")
    else:
        print("Summary CSV S3 upload failed (see logs); local CSV was still saved.")

    plot_df = rtp_per_user_per_day(user_day)
    if plot_df.empty:
        print("No base-game RTP values to plot; skipping figure.")
        raise SystemExit(0)

    local_path, uri = save_figure_local_and_s3(plot_df)
    print(f"Wrote histogram to {local_path.resolve()}")
    if uri:
        print(f"Uploaded histogram to {uri}")
    else:
        print("S3 upload failed or skipped (see logs); local file was still saved.")
