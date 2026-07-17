"""Shared helpers for the player-state HMM: feature spec, bet-day loading, churn labels.

Extracted from common-ai-research/HMM_player_states/explore_churn_states.py (the
analysis CLI modes stayed behind; this is the subset io_hmm.py and future
analyses need).

Canonical model: n_states=3 (Low / Engaged / Lapsed), Golden 7 features
(Golden 8 minus the near-constant multiplier_change_count_ratio), with the three
heavy-tailed features log1p'd on the HMM path only.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from bituslabs_ds.models.hmm_player_states.prepare_model_data import FEATURE_NAMES

# log1p only on the HMM path (these three are skew 227/144/96); the decision tree
# keeps raw X so its thresholds stay in real business units.
LOG1P_FEATURES = ["bet_amount_ratio_today_vs_history", "rtp_7_bet_days", "current_balance_max_to_avg_bet_ratio"]
# multiplier_change_count_ratio is 0 for 56%+ of rows -> covariance collapses to ~1e-7 and
# inflates the likelihood (BIC never elbows). It carries ~no signal, so drop it -> Golden 7.
DROP_FEATURES = ["multiplier_change_count_ratio"]

# State labels are tied to a seed=42 fit (state ids are not stable across refits).
STATE_LABELS = {
    3: {0: "Low", 1: "Engaged", 2: "Lapsed"},
    4: {0: "Distress", 1: "Healthy", 2: "Lapsed", 3: "Explorer"},
}


def features():
    """The Golden 7 feature names fed to the HMM (Golden 8 minus DROP_FEATURES)."""
    return [f for f in FEATURE_NAMES if f not in DROP_FEATURES]


def label(k, n_states, prefix=True):
    """Human-readable state label: 's0 Low' (prefix) or 'Low' (prefix=False); 's<k>' when unlabeled."""
    name = STATE_LABELS.get(n_states, {}).get(k)
    if name is None:
        return f"s{k}"
    return f"s{k} {name}" if prefix else name


def load_bet_days(csv_path):
    """Read the DS export (one row per (user, bet-day)), sorted by user then date.
    Adds per-user tenure `k` (1-based bet-day index) and `gap_next` (days to the next bet-day,
    NaN on the last). Returns (df, cutoff)."""
    df = pd.read_csv(csv_path, parse_dates=["bet_date"]).sort_values(["user_id", "bet_date"]).reset_index(drop=True)
    df["k"] = df.groupby("user_id").cumcount() + 1
    df["gap_next"] = df.groupby("user_id")["bet_date"].shift(-1).sub(df["bet_date"]).dt.days
    return df, df["bet_date"].max()


def _log1p_golden(X):
    """log1p the heavy-tailed Golden-7 columns in-place-safe; raises on NaN or negatives."""
    feat = features()
    if np.isnan(X).any():
        raise ValueError(f"NaN in HMM feature rows: {[feat[i] for i in np.unique(np.where(np.isnan(X))[1])]}")
    log_idx = [feat.index(f) for f in LOG1P_FEATURES]
    if (X[:, log_idx] < 0).any():
        raise ValueError(f"negative value passed to log1p in {LOG1P_FEATURES}")
    X = X.copy()
    X[:, log_idx] = np.log1p(X[:, log_idx])
    return X


def fit_hmm(X, lengths, n_states, seed=42, n_iter=200):
    m = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=n_iter, random_state=seed)
    m.fit(X, lengths)
    return m


def hmm_features(rows):
    """(X_raw, X_scaled, lengths, scaler) for a set of k>=2 rows. X_raw is untransformed (for the
    surrogate tree), X_scaled is log1p'd + standardized (for the HMM)."""
    feat = features()
    X_raw = rows[feat].to_numpy(float)
    scaler = StandardScaler().fit(_log1p_golden(X_raw))
    X_scaled = scaler.transform(_log1p_golden(X_raw))
    lengths = rows.groupby("user_id", sort=False).size().tolist()
    return X_raw, X_scaled, lengths, scaler


def decode_states(df, n_states, seed=42, train_mask=None):
    """Fit a GaussianHMM on the k>=2 rows (or, if train_mask given, the train subset of them) and
    write df['state'] for all k>=2 rows (NaN on first bet-days). Returns (model, scaler)."""
    feat = features()
    k2 = (df["k"] >= 2).to_numpy()
    rows = df.loc[k2]
    X = _log1p_golden(rows[feat].to_numpy(float))
    fit_sel = np.ones(len(rows), bool) if train_mask is None else train_mask[k2]
    scaler = StandardScaler().fit(X[fit_sel])
    Xs = scaler.transform(X)
    model = fit_hmm(Xs[fit_sel], rows[fit_sel].groupby("user_id", sort=False).size().tolist(), n_states, seed)
    df.loc[k2, "state"] = model.predict(Xs, rows.groupby("user_id", sort=False).size().tolist())
    return model, scaler


def add_leave(df, H=46, rel_mult=None):
    """Per-bet-day churn label on the user clock. leave=1 if no bet within the threshold of this bet-day;
    obs=1 if there is enough look-ahead to evaluate it. Threshold is the fixed H, unless rel_mult is set,
    then it is cadence-relative: max(H, rel_mult * the user's median inter-bet gap) -- so a naturally
    low-frequency player is not flagged for a normal-length gap."""
    cutoff = df["bet_date"].max()
    if rel_mult is None:
        thr = float(H)
    else:
        umed = df[df["k"] >= 2].groupby("user_id")["no_bet_streak_days"].median()
        thr = (rel_mult * df["user_id"].map(umed)).clip(lower=H).fillna(float(H))
    df["leave"] = (df["gap_next"].isna() | (df["gap_next"] > thr)).astype(int)
    df["obs"] = df["bet_date"] <= (cutoff - pd.to_timedelta(thr, unit="D"))
    return df


def add_cummeans(df, cols):
    """Add cumulative (expanding) per-user means `cm_<col>` = behavior as known up to each bet-day."""
    for f in cols:
        df["cm_" + f] = df.groupby("user_id")[f].expanding().mean().reset_index(level=0, drop=True)
    return df


def user_test_mask(users, frac=0.3, seed=0):
    """Boolean array over the rows (True=test), split by user_id so a user's whole sequence stays on
    one side (no leakage)."""
    rng = np.random.default_rng(seed)
    uq = users.unique()
    test = set(rng.choice(uq, size=int(frac * len(uq)), replace=False))
    return users.isin(test).to_numpy()


def eta_squared(X, labels):
    """Per-feature SS_between / SS_total -- how much of each feature's variance the state labels
    explain (0..1). X should be standardized so features are comparable."""
    grand = X.mean(0)
    sst = ((X - grand) ** 2).sum(0)
    ssb = np.zeros(X.shape[1])
    for k in np.unique(labels):
        Xk = X[labels == k]
        ssb += len(Xk) * (Xk.mean(0) - grand) ** 2
    return ssb / sst
