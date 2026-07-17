"""Input-Output HMM for player lifecycle + absorbing churn (STOP).

Markov-valid by construction:
  - Emissions are behavior-only and input-independent:  P(x_t | z_t=i) = N(x_t; mu_i, Sigma_i).
  - Transitions are input-driven (the IO part):         P(z_t=j | z_{t-1}=i, u_t) = softmax_j(W[i,j] . u_t),
    u_t = [1, log tenure, log cumulative bet] (intercept + standardized drivers).
  - One absorbing STOP state, appended only for churned users (trailing silence > H); modeled as
    OBSERVED via a posterior clamp through the emission (B[real, STOP]=0, B[stop_token, STOP]=1).
    No delta-emission, so no zero-variance collapse.

Pipeline (matches the agreed plan):
  Step 1   naive behavioral HMM  -> emission means/covars (warm start)
  Step 2   Viterbi pseudo-labels -> initial transient transition intercepts (= naive transmat)
  Step 3   (warmstart() also fits per-prev-state MLR for inspection)
  Step 4   stitch absorbing STOP: STOP row frozen (P(STOP|STOP)=1), hazard column free (init +log_t hint)
  Step 5   custom EM: forward-backward with time-varying A_t + gradient M-step for W (STOP frozen)

fit_iohmm returns {mu, var, W, pi, labels, artifact}. A full fit writes <csv>_iohmm_labels.csv (one row per
(user, bet-day): stage, p_stop = one-step churn hazard P(next=STOP | stage, that day's tenure & cum bet),
risk tier low/med/high) AND <csv>_iohmm_model.json (the deployable artifact: params + normalization constants
+ tier cuts, loaded by io_hmm_infer.py for online scoring). A user's current label = their latest bet-day row.

Run:  python io_hmm.py <csv> <warmstart|fit|calibrate> [n_behavior] [max_users]
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bituslabs_ds.models.hmm_player_states.churn_helpers import (
    LOG1P_FEATURES,
    _log1p_golden,
    add_leave,
    features,
    fit_hmm,
    label,
    load_bet_days,
)

INPUT_COLS = ["log_k", "log_cum_bet"]
VAR_FLOOR = 1e-3
# Risk tiers: high = the top HIGH_SHARE of current labels by p_stop (a population-share / intervention-budget
# cut); med floor = where the ACTUAL 46-day leave rate (calibrated on observable bet-days) reaches MED_RATE;
# low = below that. med's upper edge is the high cut, so med spans [MED_RATE leave, high cut).
HIGH_SHARE = 0.15
MED_RATE = 0.30


# --------------------------------------------------------------------------- warm-start (Steps 1-3, for inspection)


def warmstart(csv_path, n_behavior=3, seed=42):
    """Steps 1-3: behavioral HMM emissions + Viterbi labels + per-prev-state MLR(u_t -> z_t)."""
    df, _ = load_bet_days(csv_path)
    df["cum_bet"] = df.groupby("user_id")["bet_amount_today"].cumsum()
    d = df[df["k"] >= 2].copy()
    Xs = StandardScaler().fit_transform(_log1p_golden(d[features()].to_numpy(float)))
    lengths = d.groupby("user_id", sort=False).size().tolist()
    m = fit_hmm(Xs, lengths, n_behavior, seed)
    d["state"] = m.predict(Xs, lengths)
    d["log_k"] = np.log1p(d["k"])
    d["log_cum_bet"] = np.log1p(d["cum_bet"])
    d["prev"] = d.groupby("user_id")["state"].shift(1)
    tr = d.dropna(subset=["prev"])
    tr_prev = tr["prev"].astype(int)
    Xu = (tr[INPUT_COLS] - tr[INPUT_COLS].mean()) / tr[INPUT_COLS].std()
    print("emission means_:")
    print(
        pd.DataFrame(
            m.means_,
            columns=[c[:11] for c in features()],
            index=[label(i, n_behavior, prefix=False) for i in range(n_behavior)],
        )
        .round(2)
        .to_string()
    )
    print("\nunconditional transmat %:")
    print((100 * pd.DataFrame(m.transmat_)).round(1).to_string())
    for i in range(n_behavior):
        clf = LogisticRegression(max_iter=2000).fit(Xu[tr_prev == i], tr.loc[tr_prev == i, "state"])
        print(f"\nMLR from {label(i, n_behavior, prefix=False)}:  coef(log_k, log_cum_bet) per dest")
        print(pd.DataFrame(clf.coef_, columns=INPUT_COLS, index=[f"->s{c}" for c in clf.classes_]).round(3).to_string())


# --------------------------------------------------------------------------- IO-HMM pieces


def _build_data(csv_path, n_behavior, H, seed, max_users=None):
    """Return augmented sequences (STOP appended for churned users) + warm-start parameters.
    Xs: scaled behavior (STOP rows = 0, clamped); U: [1, log_k_z, log_cum_bet_z] (STOP row = last real u);
    is_stop, starts, lengths; warm mu0, var0, W0; and the real-row frame d for reporting."""
    df, cutoff = load_bet_days(csv_path)
    df["cum_bet"] = df.groupby("user_id")["bet_amount_today"].cumsum()
    add_leave(df, H)
    d = df[df["k"] >= 2].copy().reset_index(drop=True)
    if max_users is not None:
        keep = d["user_id"].unique()[:max_users]
        d = d[d["user_id"].isin(keep)].reset_index(drop=True)

    feat = features()
    scaler = StandardScaler().fit(_log1p_golden(d[feat].to_numpy(float)))
    sb = scaler.transform(_log1p_golden(d[feat].to_numpy(float)))  # (n_real, D) scaled behavior
    lengths_real = d.groupby("user_id", sort=False).size().tolist()
    m0 = fit_hmm(sb, lengths_real, n_behavior, seed)  # Step 1+2
    mu0 = m0.means_.copy()
    var0 = np.maximum(np.array([np.diag(c) for c in m0.covars_]), VAR_FLOOR)

    d["log_k"] = np.log1p(d["k"])
    d["log_cum_bet"] = np.log1p(d["cum_bet"])
    uraw = d[INPUT_COLS].to_numpy(float)
    u_mean, u_std = uraw.mean(0), uraw.std(0)  # saved for inference (no refit at serve time)
    uz = (uraw - u_mean) / u_std
    last_date = d.groupby("user_id")["bet_date"].transform("max")
    churned = ((cutoff - last_date).dt.days > H).to_numpy()

    uids = d["user_id"].to_numpy()
    _, first, counts = np.unique(uids, return_index=True, return_counts=True)
    order = np.argsort(first)  # users in first-appearance (sorted) order
    first, counts = first[order], counts[order]
    D, K = sb.shape[1], n_behavior + 1
    N = len(d) + int(churned[first].sum())  # +1 per churned user
    Xs = np.zeros((N, D))
    U = np.zeros((N, 3))
    is_stop = np.zeros(N, bool)
    real_idx = np.empty(len(d), int)  # map real row -> position in augmented array
    starts, lengths = [], []
    pos = 0
    for s, c in zip(first, counts):
        rows = slice(s, s + c)
        starts.append(pos)
        Xs[pos : pos + c] = sb[rows]
        U[pos : pos + c, 0] = 1.0
        U[pos : pos + c, 1:] = uz[rows]
        real_idx[s : s + c] = np.arange(pos, pos + c)
        pos += c
        if churned[s]:  # append STOP token
            is_stop[pos] = True
            U[pos, 0] = 1.0
            U[pos, 1:] = uz[s + c - 1]  # input = last real bet-day's drivers
            pos += 1
        lengths.append(c + (1 if churned[s] else 0))

    # warm transition weights: intercepts = log naive transmat; STOP column = low base + +log_k hint
    W0 = np.zeros((K, K, 3))
    W0[:n_behavior, :n_behavior, 0] = np.log(m0.transmat_ + 1e-6)
    W0[:n_behavior, K - 1, 0] = np.log(0.05)  # base ~5% per-step STOP
    W0[:n_behavior, K - 1, 1] = 0.5  # hint: higher tenure -> STOP
    pi = np.zeros(K)
    pi[:n_behavior] = (
        m0.get_stationary_distribution()
        if hasattr(m0, "get_stationary_distribution")
        else np.bincount(m0.predict(sb, lengths_real), minlength=n_behavior) / len(sb)
    )
    pi = pi / pi.sum()
    return dict(
        Xs=Xs,
        U=U,
        is_stop=is_stop,
        starts=np.array(starts),
        lengths=np.array(lengths),
        mu=mu0,
        var=var0,
        W=W0,
        pi=pi,
        K=K,
        D=D,
        d=d,
        real_idx=real_idx,
        feat_mean=scaler.mean_,
        feat_scale=scaler.scale_,
        u_mean=u_mean,
        u_std=u_std,
    )


def _emission_B(Xs, mu, var, is_stop, K):
    N = len(Xs)
    nb = K - 1
    B = np.zeros((N, K))
    real = ~is_stop
    Xr = Xs[real]
    logB = np.empty((Xr.shape[0], nb))
    for j in range(nb):
        diff = Xr - mu[j]
        logB[:, j] = -0.5 * (np.log(2 * np.pi * var[j]).sum() + (diff * diff / var[j]).sum(1))
    logB -= logB.max(1, keepdims=True)  # per-row constant cancels in gamma/xi
    B[real, :nb] = np.exp(logB)
    B[is_stop, K - 1] = 1.0
    return B


def _transition_A(U, W, K):
    logits = np.einsum("tp,ijp->tij", U, W)  # A[t,i,j] = softmax_j(U_t . W[i,j])
    logits -= logits.max(2, keepdims=True)
    A = np.exp(logits)
    A /= A.sum(2, keepdims=True)
    A[:, K - 1, :] = 0.0
    A[:, K - 1, K - 1] = 1.0  # STOP absorbing
    return A


def _row_hazard(U_rows, stages, W, K):
    """Per-bet-day one-step churn hazard: P(next = STOP | current stage, this row's input u).
    U_rows (n,3), stages (n,) in 0..K-2. Returns (n,) = softmax_j(W[stage] . u)[STOP]."""
    logits = np.einsum("nkp,np->nk", W[stages], U_rows)  # (n, K)
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(1, keepdims=True)
    return p[:, K - 1]


def _forward_backward(B, A, pi, starts, lengths):
    N, K = B.shape
    alpha = np.zeros((N, K))
    beta = np.zeros((N, K))
    c = np.zeros(N)
    loglik = 0.0
    for s, L in zip(starts, lengths):
        a = pi * B[s]
        c[s] = a.sum()
        alpha[s] = a / c[s]
        for t in range(1, L):
            idx = s + t
            a = (alpha[idx - 1] @ A[idx]) * B[idx]
            c[idx] = a.sum()
            alpha[idx] = a / c[idx]
        beta[s + L - 1] = 1.0
        for t in range(L - 2, -1, -1):
            idx = s + t
            beta[idx] = (A[idx + 1] @ (B[idx + 1] * beta[idx + 1])) / c[idx + 1]
        loglik += np.log(c[s : s + L]).sum()
    gamma = alpha * beta
    gamma /= gamma.sum(1, keepdims=True)
    # xi[idx,i,j] = alpha[idx-1,i] A[idx,i,j] B[idx,j] beta[idx,j] / c[idx]   for non-start positions
    xi = np.zeros((N, K, K))
    is_start = np.zeros(N, bool)
    is_start[starts] = True
    nz = ~is_start
    idx = np.where(nz)[0]
    xi[idx] = (alpha[idx - 1][:, :, None] * A[idx] * (B[idx] * beta[idx])[:, None, :]) / c[idx][:, None, None]
    return gamma, xi, loglik


def _tier_cuts(obs_p, obs_leave, cur_p, med_rate=MED_RATE, high_share=HIGH_SHARE):
    """Derive risk-tier p_stop cuts. med floor = where the actual 46d leave rate reaches med_rate (isotonic
    inversion on observable bet-days); high = top `high_share` of current labels by p_stop. Prints the
    calibration table and validates each resulting tier (current-label share + actual leave rate)."""
    p, y, cur = np.asarray(obs_p, float), np.asarray(obs_leave, float), np.asarray(cur_p, float)
    tab = pd.DataFrame({"p_stop": p, "leave": y})
    tab["bin"] = pd.qcut(tab["p_stop"], 10, duplicates="drop")
    g = tab.groupby("bin", observed=True).agg(
        n=("leave", "size"), p_stop_mean=("p_stop", "mean"), leave_rate=("leave", "mean")
    )
    print("calibration (p_stop decile -> actual 46d leave rate):")
    print(g.assign(p_stop_mean=g["p_stop_mean"].round(3), leave_rate=(100 * g["leave_rate"]).round(1)).to_string())
    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    grid = np.linspace(p.min(), p.max(), 2000)
    r = iso.predict(grid)
    med_cut = float(grid[np.searchsorted(r, med_rate)]) if r.max() >= med_rate else float(grid[-1])
    high_cut = max(float(np.quantile(cur, 1 - high_share)), med_cut + 1e-6)
    print(
        f"\ncuts: med (leave>={med_rate:.0%}) p_stop>={med_cut:.3f} | high (top {high_share:.0%} of current labels) p_stop>={high_cut:.3f}"
    )
    for name, lo, hi in [("high", high_cut, 2.0), ("med", med_cut, high_cut), ("low", -1.0, med_cut)]:
        share = 100 * np.mean((cur >= lo) & (cur < hi))
        oseg = (p >= lo) & (p < hi)
        lr = 100 * y[oseg].mean() if oseg.any() else float("nan")
        print(f"  {name:<4} current-label {share:>4.1f}% | actual 46d leave {lr:>4.1f}%")
    return med_cut, high_cut


def calibrate_tiers(csv_path, H=46):
    """Recalibrate the risk tiers on an existing <csv>_iohmm_labels.csv without re-fitting: join the actual
    leave label, derive leave-rate-based p_stop cuts, rewrite the `risk` column and the file."""
    out = csv_path.replace(".csv", "_iohmm_labels.csv")
    labels = pd.read_csv(out, parse_dates=["bet_date"])
    df, _ = load_bet_days(csv_path)
    add_leave(df, H)
    m = labels.drop(columns=["risk"]).merge(
        df[["user_id", "bet_date", "leave", "obs"]], on=["user_id", "bet_date"], how="left"
    )
    o = m[m["obs"]]
    cur = m.sort_values("bet_date").groupby("user_id").tail(1)  # current label = each user's latest bet-day
    print(f"calibrating on {len(o)} observable bet-days (of {len(m)}); base leave {100*o['leave'].mean():.1f}%\n")
    cut_med, cut_high = _tier_cuts(o["p_stop"], o["leave"], cur["p_stop"])
    m["risk"] = pd.cut(m["p_stop"], [-1, cut_med, cut_high, 2.0], labels=["low", "med", "high"])
    m[["user_id", "bet_date", "k", "stage", "p_stop", "risk"]].to_csv(out, index=False)
    latest = m.sort_values("bet_date").groupby("user_id").tail(1)
    print(f"\nrewrote {out}. current-label tier mix (each user's latest bet-day):")
    print((100 * latest["risk"].value_counts(normalize=True).reindex(["high", "med", "low"])).round(1).to_string())


def fit_iohmm(csv_path, n_behavior=3, H=46, seed=42, em_iters=20, inner=20, lr=1.0, reg=1e-3, max_users=None):
    dat = _build_data(csv_path, n_behavior, H, seed, max_users)
    Xs, U, is_stop, starts, lengths = dat["Xs"], dat["U"], dat["is_stop"], dat["starts"], dat["lengths"]
    mu, var, W, pi, K = dat["mu"], dat["var"], dat["W"], dat["pi"], dat["K"]
    nb = K - 1
    real = ~is_stop
    print(f"IO-HMM EM: {len(starts)} users, {len(Xs)} obs ({is_stop.sum()} STOP tokens), K={K} (nb={nb}+STOP)\n")
    prev_ll = None
    for it in range(em_iters):
        B = _emission_B(Xs, mu, var, is_stop, K)
        A = _transition_A(U, W, K)
        gamma, xi, ll = _forward_backward(B, A, pi, starts, lengths)
        # ll is a per-row-normalized (relative) log-lik monitor for convergence, not the absolute model
        # log-lik: _emission_B subtracts each row's max log-density (offset cancels in gamma/xi, so the
        # fitted params are unaffected). Read it for monotonicity / delta->0, not as an absolute value.
        print(f"  EM {it:2d}  loglik={ll:.1f}" + ("" if prev_ll is None else f"  d={ll-prev_ll:+.1f}"))
        prev_ll = ll
        # M-step emissions (transient, real rows)
        for j in range(nb):
            w = gamma[real, j]
            sw = w.sum()
            mu[j] = (w[:, None] * Xs[real]).sum(0) / sw
            var[j] = np.maximum((w[:, None] * (Xs[real] - mu[j]) ** 2).sum(0) / sw, VAR_FLOOR)
        # M-step transitions: gradient ascent on weighted multinomial CE (STOP prev frozen)
        gamma_prev = xi.sum(2)  # (N,K) = gamma_{t-1}(i) at transition positions
        ntr = (~np.isin(np.arange(len(Xs)), starts)).sum()
        for _ in range(inner):
            A = _transition_A(U, W, K)
            grad = np.einsum("tij,tp->ijp", xi - gamma_prev[:, :, None] * A, U) / ntr
            W[:nb] += lr * grad[:nb] - lr * reg * W[:nb]
        # M-step pi
        pi = gamma[starts].mean(0)
        pi = pi / pi.sum()

    # ---- report ----
    A = _transition_A(U, W, K)
    gamma, _, _ = _forward_backward(_emission_B(Xs, mu, var, is_stop, K), A, pi, starts, lengths)
    d = dat["d"]
    d = d.assign(stage=gamma[dat["real_idx"]].argmax(1))
    print("\n=== stages (transient), by tenure ===")
    pc = [
        "k",
        "bet_amount_today",
        "no_bet_streak_days",
        "rtp_7_bet_days",
        "loss_streak_ratio_today",
        "current_balance_day_max",
    ]
    prof = d.groupby("stage")[pc].median()
    prof.insert(0, "size", d.groupby("stage").size())
    prof.insert(1, "leave%", (100 * d[d["obs"]].groupby("stage")["leave"].mean()).round(1))
    print(prof.sort_values("k").round(2).to_string())

    print("\n=== input-driven churn hazard  P(stage -> STOP | tenure)  at log_cum_bet=median ===")
    qs = {"low_tenure": -1.0, "med_tenure": 0.0, "high_tenure": 1.0}  # standardized log_k
    haz = {}
    for name, lk in qs.items():
        u = np.array([[1.0, lk, 0.0]])
        Aq = _transition_A(u, W, K)[0]
        haz[name] = [round(100 * Aq[i, K - 1], 1) for i in range(nb)]
    print(pd.DataFrame(haz, index=[f"s{i}" for i in range(nb)]).to_string())

    print("\n=== transition A at median input (transient block + ->STOP) % ===")
    Amed = _transition_A(np.array([[1.0, 0.0, 0.0]]), W, K)[0]
    print(
        pd.DataFrame(
            (100 * Amed).round(1),
            index=[f"s{i}" for i in range(K - 1)] + ["STOP"],
            columns=[f"s{i}" for i in range(K - 1)] + ["STOP"],
        )
        .iloc[:nb]
        .to_string()
    )

    # ---- per-player labels: stage + one-step churn hazard + risk tier, for every (user, bet-day) ----
    Ur = U[dat["real_idx"]]  # this row's input [1, log_k_z, log_cum_bet_z]
    d["p_stop"] = _row_hazard(Ur, d["stage"].to_numpy(), W, K)
    print("\n=== risk tier calibration ===")
    o = d[d["obs"]]
    cur = d.sort_values("bet_date").groupby("user_id").tail(1)  # current label = each user's latest bet-day
    cut_med, cut_high = _tier_cuts(o["p_stop"], o["leave"], cur["p_stop"])
    d["risk"] = pd.cut(d["p_stop"], [-1, cut_med, cut_high, 2.0], labels=["low", "med", "high"])
    labels = d[["user_id", "bet_date", "k", "stage", "p_stop", "risk"]].copy()
    latest = labels.sort_values("bet_date").groupby("user_id").tail(1)  # actionable view: each user's current label
    if max_users is None:  # only the full run writes the canonical file
        out = csv_path.replace(".csv", "_iohmm_labels.csv")
        labels.to_csv(out, index=False)
        print(f"\n=== per-player labels written -> {out}  ({len(labels)} bet-day rows, {len(latest)} users) ===")
    else:
        print(f"\n=== subset run (max_users={max_users}): labels NOT written to file (returned only) ===")
    print("current-label tier mix (each user's latest bet-day):")
    print((100 * latest["risk"].value_counts(normalize=True).reindex(["high", "med", "low"])).round(1).to_string())

    # deployable artifact: everything needed to score a player online, incl. the training normalization
    # constants (never re-fit at serve time) and the calibrated tier cuts.
    artifact = {
        "n_behavior": n_behavior,
        "H": H,
        "seed": seed,
        "stop_state": nb,
        "features": features(),
        "log1p_features": LOG1P_FEATURES,
        "feat_mean": dat["feat_mean"].tolist(),
        "feat_scale": dat["feat_scale"].tolist(),
        "u_mean": dat["u_mean"].tolist(),
        "u_std": dat["u_std"].tolist(),
        "mu": mu.tolist(),
        "var": var.tolist(),
        "W": W.tolist(),
        "pi": pi.tolist(),
        "med_cut": float(cut_med),
        "high_cut": float(cut_high),
    }
    if max_users is None:
        mpath = csv_path.replace(".csv", "_iohmm_model.json")
        with open(mpath, "w") as fh:
            json.dump(artifact, fh)
        print(f"model artifact written -> {mpath}")
    return dict(mu=mu, var=var, W=W, pi=pi, labels=labels, artifact=artifact)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/selected_hmm_features_fm01_cny.csv"
    mode = sys.argv[2] if len(sys.argv) > 2 else "fit"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    if mode == "warmstart":
        warmstart(csv, n_behavior=n)
    elif mode == "calibrate":
        calibrate_tiers(csv)
    else:
        fit_iohmm(csv, n_behavior=n, max_users=(int(sys.argv[4]) if len(sys.argv) > 4 else None))
