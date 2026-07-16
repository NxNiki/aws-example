"""Compare FM01 lifecycle IO-HMM stages with kmeans cluster labels on the daily grain.

Joins the HMM labels CSV (one row per k>=2 user-bet-day: stage, p_stop, risk)
with a cluster_labels.parquet from the cluster-analysis pipeline on
(user_id, bet_date). The inner join drops each user's first bet-day from the
kmeans side automatically (the HMM never labels k=1 rows). Reports:

* contingency matrix (counts + row/column-normalized)
* chance-corrected agreement: ARI, AMI, NMI
* empirical bet-day-to-bet-day transition matrices for both labelings
  (consecutive bet-days on the user clock, however far apart in calendar days)
* churn signal per label: mean/median p_stop and actual 46-day leave rate

Run:
  poetry run python jobs/fish_hunter/analysis_hmm_vs_kmeans.py \\
    --kmeans-labels jobs/output_fish_hunter/daily_kmeans/<run-id>/cluster_labels.parquet
"""

import argparse

import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

HMM_LABELS_DEFAULT = "jobs/output_fish_hunter/lifecycle_hmm/selected_hmm_features_fm01_cny_iohmm_labels.csv"
H_DAYS = 46


def transition_matrix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df.sort_values(["user_id", "bet_date"], kind="stable")
    prev = d.groupby("user_id")[col].shift(1)
    keep = prev.notna()
    return pd.crosstab(prev[keep].astype(int), d.loc[keep, col], normalize="index")


def add_leave(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["user_id", "bet_date"], kind="stable").copy()
    gap_next = d.groupby("user_id")["bet_date"].shift(-1).sub(d["bet_date"]).dt.days
    d["leave"] = (gap_next.isna() | (gap_next > H_DAYS)).astype(int)
    d["obs"] = d["bet_date"] <= (d["bet_date"].max() - pd.Timedelta(days=H_DAYS))
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmm-labels", default=HMM_LABELS_DEFAULT)
    parser.add_argument("--kmeans-labels", required=True, help="cluster_labels.parquet from a pipeline run")
    args = parser.parse_args()

    hmm = pd.read_csv(args.hmm_labels, parse_dates=["bet_date"])
    km = pd.read_parquet(args.kmeans_labels)
    label_col = [c for c in km.columns if c.startswith(("kmeans", "hierarchical", "dbscan"))][0]
    km = km.dropna(subset=[label_col]).copy()
    km["cluster"] = km[label_col].astype(int)
    km["bet_date"] = pd.to_datetime(km["bet_date"])

    m = hmm.merge(km[["user_id", "bet_date", "cluster"]], on=["user_id", "bet_date"], how="inner")
    print(f"hmm rows (k>=2): {len(hmm):,} | kmeans rows: {len(km):,} | matched: {len(m):,}")
    print(f"kmeans rows dropped by the join (first bet-days etc.): {len(km) - len(m):,}\n")

    ct = pd.crosstab(m["stage"], m["cluster"])
    print("=== contingency: counts (rows=HMM stage, cols=kmeans cluster) ===")
    print(ct.to_string())
    print("\n=== P(cluster | stage) % ===")
    print((100 * pd.crosstab(m["stage"], m["cluster"], normalize="index")).round(1).to_string())
    print("\n=== P(stage | cluster) % ===")
    print((100 * pd.crosstab(m["stage"], m["cluster"], normalize="columns")).round(1).to_string())

    print("\n=== chance-corrected agreement ===")
    print(f"ARI: {adjusted_rand_score(m['stage'], m['cluster']):.3f}")
    print(f"AMI: {adjusted_mutual_info_score(m['stage'], m['cluster']):.3f}")
    print(f"NMI: {normalized_mutual_info_score(m['stage'], m['cluster']):.3f}")

    print("\n=== HMM stage transitions P(next | current) % (consecutive bet-days) ===")
    print((100 * transition_matrix(m, "stage")).round(1).to_string())
    print("\n=== kmeans cluster transitions P(next | current) % (consecutive bet-days) ===")
    print((100 * transition_matrix(m, "cluster")).round(1).to_string())

    m = add_leave(m)
    o = m[m["obs"]]
    print(
        f"\n=== churn signal (observable bet-days: {len(o):,}; base {H_DAYS}d leave {100 * o['leave'].mean():.1f}%) ==="
    )
    for col in ["stage", "cluster"]:
        g = o.groupby(col).agg(
            rows=("leave", "size"),
            leave_rate=("leave", "mean"),
            p_stop_mean=("p_stop", "mean"),
            p_stop_median=("p_stop", "median"),
        )
        g["leave_rate"] = (100 * g["leave_rate"]).round(1)
        print(f"\nby {col}:")
        print(g.round(3).to_string())


if __name__ == "__main__":
    main()
