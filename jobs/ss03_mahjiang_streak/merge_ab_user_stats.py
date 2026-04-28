"""
Merge SS03 AB user mapping with dated user stats on ``user_id``.

``ss03_ab_user_id.csv`` is one row per user (``ab_group_id``, ``user_id``) with no date.
The stats CSV has ``date`` (renamed to ``activity_date``), ``type`` (mapped to ``bet_type``:
``0`` → ``BASE``, any other value → ``FREE``), ``user_id``, and metric columns. Rows are joined
on ``user_id``; dates are kept from the stats file (multiple rows per user per day are preserved).
You may also supply columns already named ``activity_date`` / ``bet_type`` instead.

Default paths point at ``~/Documents/data_ss03/``; override with CLI args if needed.
Output is written next to the inputs unless ``--output`` is set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path.home() / "Documents" / "data_ss03"
DEFAULT_AB = DEFAULT_DATA_DIR / "ss03_ab_group_users.csv"
DEFAULT_STATS = DEFAULT_DATA_DIR / "date_user_rtp_20260421-UTC-7.csv"


def _strip_ab_group_id(series: pd.Series) -> pd.Series:
    """Normalize AB ids that may be quoted like \"\"\"uuid\"\"\" in CSV."""
    s = series.astype(str)
    s = s.str.replace('"""', "", regex=False)
    return s.str.strip('"')


def load_ab_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ab_group_id" not in df.columns or "user_id" not in df.columns:
        raise ValueError(f"Expected columns ab_group_id, user_id in {path}")
    df = df.copy()
    df["ab_group_id"] = _strip_ab_group_id(cast(pd.Series, df["ab_group_id"]))
    _uid = pd.to_numeric(cast(pd.Series, df["user_id"]), errors="coerce")
    df["user_id"] = cast(pd.Series, _uid).astype("Int64")
    df = df.dropna(subset=["user_id"])
    dup = df["user_id"].duplicated(keep=False)
    if dup.any():
        df = df.drop_duplicates(subset=["user_id"], keep="first")
    return df


def load_stats_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "user_id" not in df.columns:
        raise ValueError(f"Expected column user_id in {path}")
    if "activity_date" not in df.columns and "date" not in df.columns:
        raise ValueError(f"Expected column date or activity_date in {path}")
    if "bet_type" not in df.columns and "type" not in df.columns:
        raise ValueError(f"Expected column type or bet_type in {path}")
    df = df.copy()
    if "activity_date" not in df.columns:
        df = df.rename(columns={"date": "activity_date"})
    if "bet_type" not in df.columns:
        t = pd.to_numeric(cast(pd.Series, df["type"]), errors="coerce")
        df["bet_type"] = np.where(t == 0, "BASE", "FREE")
        df = df.drop(columns=["type"])
    _uid = pd.to_numeric(cast(pd.Series, df["user_id"]), errors="coerce")
    df["user_id"] = cast(pd.Series, _uid).astype("Int64")
    df["activity_date"] = pd.to_datetime(df["activity_date"]).dt.date
    return df


def merge_ab_and_stats(ab: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Left-join stats to AB mapping on user_id; preserves all dated stat rows.

    Adds ``ab_arm`` — the first four lowercased characters of ``ab_group_id`` (e.g. ``4a04``,
    ``4f1a``) — so downstream scripts can group/label by arm without re-slicing.
    """
    merged = stats.merge(ab, on="user_id", how="left")
    if "ab_group_id" in merged.columns:
        merged["ab_arm"] = merged["ab_group_id"].astype(str).str[:4].str.lower()
    cols = list(merged.columns)
    if "ab_group_id" in cols:
        cols.remove("ab_group_id")
        insert_at = cols.index("user_id") + 1
        cols.insert(insert_at, "ab_group_id")
        if "ab_arm" in cols:
            cols.remove("ab_arm")
            cols.insert(insert_at + 1, "ab_arm")
        merged = cast(pd.DataFrame, merged[cols])
    return merged


def default_output_path(stats_path: Path) -> Path:
    stem = stats_path.stem
    return stats_path.parent / f"merged_{stem}_with_ab.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SS03 AB users and dated stats CSVs.")
    parser.add_argument(
        "--ab-csv",
        type=Path,
        default=DEFAULT_AB,
        help=f"AB mapping CSV (default: {DEFAULT_AB})",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=DEFAULT_STATS,
        help=f"Dated stats CSV (default: {DEFAULT_STATS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: same folder as stats, merged_<stem>_with_ab.csv)",
    )
    args = parser.parse_args()

    ab_path = args.ab_csv.expanduser().resolve()
    stats_path = args.stats_csv.expanduser().resolve()
    out_path = args.output.expanduser().resolve() if args.output else default_output_path(stats_path)

    if not ab_path.is_file():
        raise FileNotFoundError(ab_path)
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)

    ab = load_ab_csv(ab_path)
    stats = load_stats_csv(stats_path)
    merged = merge_ab_and_stats(ab, stats)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
