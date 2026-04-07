import argparse
import logging
import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from bituslabs_ds.config import S3_BUCKET
from bituslabs_ds.s3_utils import upload_file_to_s3

file1 = "/Users/niuxin/Documents/data_ss03/ss03_incentive_stats.csv"
file2 = "/Users/niuxin/Documents/data_ss03/ss03_incentive_user_group_by_day.csv"

OUTPUT_DIR = _root / "jobs" / "output" / "ss03_mahjiang_streak"
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

    summary = user_day.groupby(["date", "partition_ab"], as_index=False).agg(
        num_users=("user_id", "count"),
        num_users_with_fg=("has_fg", "sum"),
        num_users_low_spin=("low_spin", "sum"),
        num_users_high_rtp=("high_rtp", "sum"),
        num_users_incentivized=("has_incentivized", "sum"),
    )
    summary = summary.sort_values(["date", "partition_ab"]).reset_index(drop=True)  # pyright: ignore[reportCallIssue]

    nu = summary["num_users"].replace(0, np.nan)
    summary["ratio_users_with_fg"] = summary["num_users_with_fg"] / nu
    summary["ratio_users_low_spin"] = summary["num_users_low_spin"] / nu
    summary["ratio_users_high_rtp"] = summary["num_users_high_rtp"] / nu
    summary["ratio_users_incentivized"] = summary["num_users_incentivized"] / nu

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
