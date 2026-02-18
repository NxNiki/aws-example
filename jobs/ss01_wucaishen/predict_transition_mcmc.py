"""
Full 3×3 Markov transition model: for each origin state i, behavioral features X
determine the probability of transitioning to each target state j (9 conditions total).

Structure: 3 origin states × (3-1) target coefficients = separate beta per row.
  P(S_{t+1}=j | S_t=i, X) = softmax(eta)[j],  eta = X @ beta[i]
  beta shape: (3, n_features, 2) — one coefficient set per origin state.

Optional spike-and-slab yields Posterior Inclusion Probability (PIP) per (origin, feature, target).

Usage:
  Local: run after analysis_cluster_transition_stats.py (reads/writes OUTPUT_DIR).
  SageMaker: --input /opt/ml/processing/input --output /opt/ml/processing/output

Outputs (all written to --output, which SageMaker uploads to S3):
  - transition_mcmc_report.md          (markdown report)
  - transition_mcmc_coefficient_summary.csv
  - transition_mcmc_run_info.csv       (n_obs, n_chains, duration_sec, etc.)
  - transition_mcmc_diagnostics.csv   (rhat, ess_bulk, ess_tail per coefficient)
"""

import argparse
import sys
import time
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

# Allow running as script (e.g. python predict_transition_mcmc.py) from any cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from jobs.ss01_wucaishen.cluster_transition_data import ensure_merged_parquet

# Same output dir as analysis_cluster_transition_stats
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_ss01_cluster_transition"
FEATURE_START_INDEX = 6
MAX_FEATURES_FOR_MCMC = 25  # cap predictors so MCMC is tractable
N_SAMPLES = 500  # MCMC draws (tune + draw)
N_TUNE = 500
RANDOM_SEED = 42
HDI_PROB = 0.95  # credible interval for "predictive" variable
# Spike-and-slab: if True, compute Posterior Inclusion Probability (PIP) per predictor
USE_SPIKE_SLAB = True
PIP_THRESHOLD = 0.5  # predictors with PIP > this are considered "included"
# Parallel MCMC: chains run in parallel across cores; set to 1 to disable
N_CORES = min(4, max(1, (cpu_count() or 2) - 1))  # leave one core free by default
N_CHAINS = max(2, N_CORES)  # at least 2 chains for diagnostics


def _is_numeric_column(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    return pd.api.types.is_numeric_dtype(df[col])


def load_merged_and_prepare(
    merged_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str], list[int], int]:
    """Load merged parquet; build X (features only), y (next_cluster), current_state_idx.
    Returns (merged, X, y, current_state_idx, predictor_names, class_list, n_origin_states).
    Current state selects which row of beta is used (origin-specific coefficients).
    """
    merged = pd.read_parquet(merged_path)
    all_cols = list(merged.columns)
    feature_cands = [c for i, c in enumerate(all_cols) if i >= FEATURE_START_INDEX and _is_numeric_column(merged, c)][
        :MAX_FEATURES_FOR_MCMC
    ]

    current_state_idx = merged["cluster_label"].astype(int).values
    n_origin_states = int(current_state_idx.max()) + 1
    # Features only (no current-state dummies); current state indexes beta
    X = np.asarray(merged[feature_cands].fillna(merged[feature_cands].median()).values, dtype=np.float64)
    mean_f = np.nanmean(X, axis=0)
    std_f = np.nanstd(X, axis=0)
    std_f[std_f == 0] = 1.0
    X = (X - mean_f) / std_f
    X = np.asarray(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)

    y = merged["next_cluster"].astype(int).values
    predictor_names = list(feature_cands)
    class_list = list(range(n_origin_states))
    if not (np.isfinite(X).all() and np.isfinite(y).all() and np.isfinite(current_state_idx).all()):
        raise ValueError("X, y, or current_state_idx contains non-finite values after preprocessing.")
    return merged, X, y, current_state_idx, predictor_names, class_list, n_origin_states


def build_model_and_sample(
    X: np.ndarray,
    y: np.ndarray,
    current_state_idx: np.ndarray,
    n_origin_states: int,
    n_classes: int,
    random_seed: int,
    use_spike_slab: bool = True,
):
    """Full 3×3 Markov transition model: beta shape (n_origin_states, n_features, 2).
    For each row, eta = X @ beta[current_state_idx]; then softmax gives P(next state).
    """
    n_samples, n_features = X.shape
    n_eta = n_classes - 1  # 2 for 3 states (reference = 0)

    with pm.Model() as model:
        # Index into beta by current state (passed as data)
        idx = pm.Data("current_state_idx", current_state_idx, mutable=False)
        if use_spike_slab:
            gamma = pm.Bernoulli("gamma", p=0.5, shape=(n_origin_states, n_features, n_eta))
            b = pm.Normal("b", mu=0, sigma=1.5, shape=(n_origin_states, n_features, n_eta))
            beta = pm.Deterministic("beta", gamma * b)
        else:
            beta = pm.Normal("beta", mu=0, sigma=1.5, shape=(n_origin_states, n_features, n_eta))
        # beta[idx] -> (n_samples, n_features, 2); then eta = (X * beta_current).sum(axis=1) -> (n_samples, 2)
        beta_current = beta[idx]  # (n_samples, n_features, n_eta)
        eta = pm.math.sum(X[:, :, None] * beta_current, axis=1)  # (n_samples, n_eta)
        p_ref = pm.math.concatenate([pm.math.zeros((n_samples, 1)), eta], axis=1)
        p_ref_stable = p_ref - pm.math.max(p_ref, axis=1, keepdims=True)
        p_raw = pm.math.softmax(p_ref_stable)
        p = pm.math.clip(p_raw, 1e-7, 1.0) / pm.math.sum(pm.math.clip(p_raw, 1e-7, 1.0), axis=1, keepdims=True)
        y_obs = pm.Categorical("y_obs", p=p, observed=y)
        idata = pm.sample(
            draws=N_SAMPLES,
            tune=N_TUNE,
            chains=N_CHAINS,
            cores=N_CORES,
            random_seed=random_seed,
            idata_kwargs={"log_likelihood": True},
            progressbar=True,
            target_accept=0.9,
        )
    return idata


def summarize_predictive_variables(
    idata,
    predictor_names: list[str],
    n_origin_states: int,
    n_classes: int,
    hdi_prob: float = 0.95,
    include_pip: bool = False,
) -> pd.DataFrame:
    """From MCMC trace (beta shape n_origin x n_features x n_eta), compute summary and PIP."""
    n_eta = n_classes - 1
    beta_post = idata.posterior["beta"].values  # (chain, draw, n_origin, n_features, n_eta)
    beta_flat = beta_post.reshape(-1, n_origin_states, len(predictor_names), n_eta)
    pip_flat = None
    if include_pip and "gamma" in idata.posterior:
        gamma_post = idata.posterior["gamma"].values
        pip_flat = gamma_post.reshape(-1, n_origin_states, len(predictor_names), n_eta)

    rows = []
    for i_origin in range(n_origin_states):
        for j in range(len(predictor_names)):
            for k in range(n_eta):
                beta_ijk = beta_flat[:, i_origin, j, k]
                mean = float(np.mean(beta_ijk))
                sd = float(np.std(beta_ijk))
                hdi_lo, hdi_hi = az.hdi(beta_ijk, hdi_prob=hdi_prob)
                hdi_lo, hdi_hi = float(hdi_lo), float(hdi_hi)
                excludes_zero = hdi_lo > 0 or hdi_hi < 0
                row = {
                    "origin_state": i_origin,
                    "predictor": predictor_names[j],
                    "target_class": k + 1,  # 1 or 2 (0 is reference)
                    "mean": mean,
                    "sd": sd,
                    "hdi_low": hdi_lo,
                    "hdi_high": hdi_hi,
                    "hdi_excludes_zero": excludes_zero,
                }
                if pip_flat is not None:
                    row["pip"] = float(np.mean(pip_flat[:, i_origin, j, k]))
                rows.append(row)
    return pd.DataFrame(rows)


def build_diagnostics_df(
    idata,
    predictor_names: list[str],
    n_origin_states: int,
    n_eta: int,
) -> pd.DataFrame:
    """Build a CSV of MCMC diagnostics (rhat, ess_bulk, ess_tail) per (origin, predictor, target)."""
    rhat = az.rhat(idata, var_names=["beta"]).values  # (n_origin, n_features, n_eta)
    ess_bulk = az.ess(idata, var_names=["beta"], method="bulk").values
    ess_tail = az.ess(idata, var_names=["beta"], method="tail").values
    rows = []
    for i_origin in range(n_origin_states):
        for j in range(len(predictor_names)):
            for k in range(n_eta):
                rows.append(
                    {
                        "origin_state": i_origin,
                        "predictor": predictor_names[j],
                        "target_class": k + 1,
                        "rhat": float(rhat[i_origin, j, k]),
                        "ess_bulk": float(ess_bulk[i_origin, j, k]),
                        "ess_tail": float(ess_tail[i_origin, j, k]),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    summary_df: pd.DataFrame,
    merged: pd.DataFrame,
    predictor_names: list[str],
    class_list: list[int],
    n_origin_states: int,
    output_dir: Path,
    use_spike_slab: bool,
    hdi_prob: float = 0.95,
    pip_threshold: float = 0.5,
) -> Path:
    """Write markdown report: full 3×3 transition matrix model and origin-specific predictor summary."""
    out_path = output_dir / "transition_mcmc_report.md"
    has_pip = "pip" in summary_df.columns

    with open(out_path, "w") as f:
        f.write("# Full 3×3 Markov Transition Model: Predictors of State Change\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write("## 1. Model\n\n")
        f.write("We model all **9 transition probabilities** (3 origin states × 3 target states). ")
        f.write("Each **row** of the transition matrix has its own set of coefficients:\n\n")
        f.write(
            "$$P(S_{t+1}=j \\mid S_t=i, \\mathbf{X}) = \\frac{\\exp(\\eta_j)}{\\sum_{k=0}^{2} \\exp(\\eta_k)}, \\quad \\eta = \\mathbf{X} \\beta_i$$\n\n"
        )
        f.write("- \\(i\\) = current (origin) state; \\(j\\) = next (target) state.\n")
        f.write("- \\(\\mathbf{X}\\): standardized behavioral features (no state dummies).\n")
        f.write("- \\(\\beta_i\\): coefficient vector **for origin state \\(i\\)**; shape (3, n_features, 2). ")
        f.write(
            "So the same feature can have different effects depending on where the user is coming from (e.g. big loss from Low-Risk vs from High-Risk).\n"
        )
        f.write("- **Reference target** is state 0; \\(\\eta\\) has 2 components for targets 1 and 2.\n\n")
        if use_spike_slab and has_pip:
            f.write("### Variable selection (spike-and-slab)\n\n")
            f.write("Spike-and-slab gives a **Posterior Inclusion Probability (PIP)** per (origin, feature, target). ")
            f.write(f"PIP > {pip_threshold} indicates the feature is likely a predictor for that transition.\n\n")
        f.write("## 2. Data\n\n")
        f.write(f"- **Rows (transitions):** {len(merged):,}\n")
        f.write(f"- **Features:** {len(predictor_names)}\n")
        f.write(f"- **Origin states:** 0..{n_origin_states - 1}; **Target states:** {class_list}\n\n")
        f.write("## 3. Coefficient summary (by origin state)\n\n")
        if has_pip:
            f.write("| Origin | Predictor | Target | Mean(β) | SD | HDI low | HDI high | HDI excl. 0 | **PIP** |\n")
            f.write("|--------|-----------|--------|--------|-----|---------|----------|-------------|--------|\n")
            for _, r in summary_df.iterrows():
                pip_str = f"{r['pip']:.3f}" if pd.notna(r.get("pip")) else "—"
                f.write(
                    f"| {r['origin_state']} | {r['predictor']} | {r['target_class']} | {r['mean']:.4f} | {r['sd']:.4f} | {r['hdi_low']:.4f} | {r['hdi_high']:.4f} | {r['hdi_excludes_zero']} | **{pip_str}** |\n"
                )
        else:
            f.write("| Origin | Predictor | Target | Mean(β) | SD | HDI low | HDI high | HDI excl. 0 |\n")
            f.write("|--------|-----------|--------|--------|-----|---------|----------|-------------|\n")
            for _, r in summary_df.iterrows():
                f.write(
                    f"| {r['origin_state']} | {r['predictor']} | {r['target_class']} | {r['mean']:.4f} | {r['sd']:.4f} | {r['hdi_low']:.4f} | {r['hdi_high']:.4f} | {r['hdi_excludes_zero']} |\n"
                )
        f.write("\n## 4. Predictors associated with transitions\n\n")
        if has_pip:
            inc = summary_df[summary_df["pip"] > pip_threshold]
            if len(inc) > 0:
                for orig in sorted(summary_df["origin_state"].unique()):
                    sub = inc[inc["origin_state"] == orig]
                    if len(sub) == 0:
                        continue
                    f.write(f"**From state {orig}:** ")
                    f.write(", ".join(sub["predictor"].unique().tolist()) + "\n\n")
            else:
                f.write("No predictor had PIP above the threshold.\n")
        else:
            sig = summary_df[summary_df["hdi_excludes_zero"]]
            if len(sig) > 0:
                for orig in sorted(summary_df["origin_state"].unique()):
                    sub = sig[sig["origin_state"] == orig]
                    if len(sub) > 0:
                        f.write(f"**From state {orig}:** " + ", ".join(sub["predictor"].unique().tolist()) + "\n\n")
            else:
                f.write("No predictor had HDI excluding zero.\n")
        f.write("\n## 5. Notes\n\n")
        f.write(
            "- **Full transition matrix**: Each origin state has its own \\(\\beta_i\\); total 3 × n_features × 2 parameters.\n"
        )
        f.write(
            "- **Multi-user**: Global coefficients only; hierarchical (user-level) random effects could be added.\n"
        )
    return out_path


def _parse_args():
    parser = argparse.ArgumentParser(description="MCMC transition model (3×3 Markov).")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input path: directory containing merged_with_transitions.parquet, or path to the parquet file. Default: OUTPUT_DIR.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for CSV and report. Default: OUTPUT_DIR.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.input is not None:
        inp = Path(args.input)
        if inp.suffix == ".parquet":
            merged_path = inp
        else:
            merged_path = inp / "merged_with_transitions.parquet"
    else:
        merged_path = OUTPUT_DIR / "merged_with_transitions.parquet"
        # Local run: ensure merged parquet exists (from cache or build from S3 + features)
        if not merged_path.exists():
            ensure_merged_parquet(merged_path)
    if not merged_path.exists():
        print(f"Merged data not found at {merged_path}")
        return

    out_dir = Path(args.output) if args.output is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading merged data (features + origin/target state)...")
    merged, X, y, current_state_idx, predictor_names, class_list, n_origin_states = load_merged_and_prepare(merged_path)
    n_classes = len(class_list)
    print(
        f"Design: {X.shape[0]} rows, {X.shape[1]} features. Origin states: 0..{n_origin_states-1}, targets: {class_list} (9 transition conditions)"
    )

    print(
        f"Running MCMC (3×3 Markov"
        + (" + spike-and-slab" if USE_SPIKE_SLAB else "")
        + f") with {N_CHAINS} chains on {N_CORES} cores..."
    )
    t0 = time.perf_counter()
    idata = build_model_and_sample(
        X,
        y,
        current_state_idx,
        n_origin_states,
        n_classes,
        RANDOM_SEED,
        use_spike_slab=USE_SPIKE_SLAB,
    )
    duration_sec = time.perf_counter() - t0
    print(f"MCMC finished in {duration_sec:.1f} s")

    # Run info and diagnostics (saved to out_dir → S3 when on SageMaker)
    run_info = pd.DataFrame(
        [
            {
                "n_observations": len(merged),
                "n_features": len(predictor_names),
                "n_origin_states": n_origin_states,
                "n_classes": n_classes,
                "n_chains": N_CHAINS,
                "n_draws": N_SAMPLES,
                "n_tune": N_TUNE,
                "duration_sec": round(duration_sec, 2),
                "timestamp": datetime.now().isoformat(),
                "random_seed": RANDOM_SEED,
                "use_spike_slab": USE_SPIKE_SLAB,
            }
        ]
    )
    run_info_path = out_dir / "transition_mcmc_run_info.csv"
    run_info.to_csv(run_info_path, index=False)
    print(f"Saved run info to {run_info_path}")

    n_eta = n_classes - 1
    diagnostics_df = build_diagnostics_df(idata, predictor_names, n_origin_states, n_eta)
    diagnostics_path = out_dir / "transition_mcmc_diagnostics.csv"
    diagnostics_df.to_csv(diagnostics_path, index=False)
    print(f"Saved MCMC diagnostics to {diagnostics_path}")

    print("Summarizing posteriors (per origin state, feature, target)...")
    summary_df = summarize_predictive_variables(
        idata,
        predictor_names,
        n_origin_states,
        n_classes,
        hdi_prob=HDI_PROB,
        include_pip=USE_SPIKE_SLAB,
    )
    summary_path = out_dir / "transition_mcmc_coefficient_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved coefficient summary to {summary_path}")

    if "pip" in summary_df.columns:
        predictive = summary_df[summary_df["pip"] > PIP_THRESHOLD]["predictor"].unique().tolist()
        print(f"\nPredictors with PIP > {PIP_THRESHOLD} (any origin/target):")
    else:
        predictive = summary_df[summary_df["hdi_excludes_zero"]]["predictor"].unique().tolist()
        print(f"\nPredictors with {HDI_PROB*100:.0f}% HDI excluding 0 (any origin/target):")
    if predictive:
        for p in predictive:
            print(f"  - {p}")
    else:
        print("  (none)")

    report_path = write_report(
        summary_df,
        merged,
        predictor_names,
        class_list,
        n_origin_states,
        out_dir,
        use_spike_slab=USE_SPIKE_SLAB,
        hdi_prob=HDI_PROB,
        pip_threshold=PIP_THRESHOLD,
    )
    print(f"\nReport written to {report_path}")

    return summary_df, idata


if __name__ == "__main__":
    main()
