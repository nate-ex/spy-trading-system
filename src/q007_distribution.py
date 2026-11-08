"""
Q007 — conditional forward-return DISTRIBUTION, Stage 1 (milestone 1: scale).

Estimates a conditional location--scale distribution for the forward return
`r_h` at each decision point, per horizon h, and scores it out-of-sample
with a **proper scoring rule** (Gaussian / Student-t log-likelihood) plus
calibration -- deliberately NOT R^2 of returns (see Q007's question file and
the Phase-4 brief: do not optimize for RMSE/regression accuracy).

Model (milestone 1, all-OLS, no custom optimizer):
- Location mu_h: kept minimal -- the train-set mean of `r_h`. The whole
  project's evidence (Q001/Q003-Q006) is that the conditional mean is a
  near-martingale, so a constant location is both defensible and keeps the
  scale test unconfounded by an over-fit mean. H0-location tests this
  assumption explicitly (see `q007_scale_experiment.py`).
- Scale sigma_h(X): regress |r_h - mu_h| on the scale-feature design X
  (OLS on the TRAIN fold only), predict E[|resid|] on TEST, convert to a
  standard deviation via sigma_hat = E[|resid|] * sqrt(pi/2) (the Gaussian
  relation E|Z| = sigma*sqrt(2/pi)), floored to a small positive value.
- Shape: an optional Student-t with tail dof estimated from the train-fold
  standardized residuals' excess kurtosis (method of moments, clipped).

A feature family's value is the OOS **log-score improvement** of
(baseline + family) over baseline, on a fixed common sample, validated with
PurgedWalkForwardCV + day-clustered bootstrap + day-swap placebo.

No real data is embedded here; generic enough to mirror publicly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from time_series_cv import PurgedWalkForwardCV

HORIZONS = ["5min", "10min", "15min", "30min", "to_close"]  # unchanged -- q007_experiment0/1.py
# (Q007 milestone 1, already frozen/complete) iterate this list blindly; adding "20min" here
# would silently expand a completed, pre-registered analysis's scope if ever rerun. The dashboard
# engines below only ever look up HORIZON_TARGET/HORIZON_MINUTES by explicit key, never iterate
# this list, so a new "20min" key in those two dicts (added below) is safe and doesn't need it.
HORIZON_TARGET = {
    "5min": "return_5min",
    "10min": "return_10min",
    "15min": "return_15min",
    "20min": "return_20min",
    "30min": "return_30min",
    "to_close": "return_to_close",
}
HORIZON_MINUTES = {"5min": 5, "10min": 10, "15min": 15, "20min": 20, "30min": 30}

N_SPLITS = 6
EMBARGO_FRAC = 0.01
SIGMA_FLOOR_FRAC = 0.10  # floor sigma_hat at this fraction of the train median |resid|-derived sigma


# --------------------------------------------------------------------------
# labels / purging
# --------------------------------------------------------------------------
def label_end_times(df: pd.DataFrame, horizon: str) -> pd.Series:
    """When each row's label is fully determined, for purging.

    Fixed horizons: time + h minutes. To-close: that day's 16:00 bar.
    """
    t = pd.to_datetime(df["time"]).reset_index(drop=True)
    if horizon == "to_close":
        return pd.to_datetime(df["date"].astype(str)).reset_index(drop=True) + pd.Timedelta(hours=16)
    return t + pd.Timedelta(minutes=HORIZON_MINUTES[horizon])


# --------------------------------------------------------------------------
# location-scale model (all OLS)
# --------------------------------------------------------------------------
def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coef


def _ols_predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + X @ coef[1:]


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def _t_dof_from_kurtosis(z: np.ndarray) -> float:
    """Method-of-moments Student-t dof from standardized-residual excess
    kurtosis: excess kurt = 6/(nu-4) for nu>4. Clipped to [4.5, 100]."""
    z = z[np.isfinite(z)]
    if len(z) < 50:
        return 100.0
    k = float(((z - z.mean()) ** 4).mean() / (z.var() ** 2 + 1e-30) - 3.0)
    if k <= 0.05:
        return 100.0
    nu = 6.0 / k + 4.0
    return float(np.clip(nu, 4.5, 100.0))


def fit_location_scale(r_train: np.ndarray, Xs_train: np.ndarray) -> dict:
    """Fit mu (constant = train mean) and the abs-residual scale OLS."""
    mu = float(np.mean(r_train))
    abs_resid = np.abs(r_train - mu)
    coef = _ols_fit(Xs_train, abs_resid)
    # sigma floor: fraction of the unconditional train sigma
    uncond_sigma = float(np.std(r_train)) if np.std(r_train) > 0 else 1e-9
    floor = SIGMA_FLOOR_FRAC * uncond_sigma
    # tail dof from standardized residuals under the fitted scale
    pred_abs = np.maximum(_ols_predict(coef, Xs_train), 1e-12)
    sigma_tr = np.maximum(pred_abs * np.sqrt(np.pi / 2.0), floor)
    dof = _t_dof_from_kurtosis((r_train - mu) / sigma_tr)
    return {"mu": mu, "coef": coef, "floor": floor, "dof": dof}


def predict_sigma(params: dict, Xs_test: np.ndarray) -> np.ndarray:
    pred_abs = np.maximum(_ols_predict(params["coef"], Xs_test), 1e-12)
    return np.maximum(pred_abs * np.sqrt(np.pi / 2.0), params["floor"])


# --------------------------------------------------------------------------
# proper scoring rules (log-likelihood) and calibration
# --------------------------------------------------------------------------
def gaussian_loglik(r: np.ndarray, mu: float, sigma: np.ndarray) -> np.ndarray:
    return -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((r - mu) ** 2) / (sigma ** 2)


def student_t_loglik(r: np.ndarray, mu: float, sigma: np.ndarray, dof: float) -> np.ndarray:
    """Student-t with variance matched to sigma^2 (scale s = sigma*sqrt((nu-2)/nu))."""
    from scipy.special import gammaln

    if dof >= 100:
        return gaussian_loglik(r, mu, sigma)
    s = sigma * np.sqrt((dof - 2.0) / dof)
    z = (r - mu) / s
    c = gammaln((dof + 1) / 2) - gammaln(dof / 2) - 0.5 * np.log(dof * np.pi) - np.log(s)
    return c - (dof + 1) / 2 * np.log1p(z ** 2 / dof)


def gaussian_cdf(r: np.ndarray, mu: float, sigma: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    return norm.cdf((r - mu) / sigma)


def prob_abs_gt(mu: float, sigma: np.ndarray, tau: float) -> np.ndarray:
    """Model P(|r| > tau) under Normal(mu, sigma)."""
    from scipy.stats import norm

    return norm.sf((tau - mu) / sigma) + norm.cdf((-tau - mu) / sigma)


# --------------------------------------------------------------------------
# cross-validated OOS scoring for one scale-feature set
# --------------------------------------------------------------------------
def cv_logscore(df: pd.DataFrame, horizon: str, sigma_cols: list,
                n_splits: int = N_SPLITS, embargo_frac: float = EMBARGO_FRAC) -> dict:
    """Run purged walk-forward CV; return per-test-row OOS Gaussian & t
    log-scores, the predicted (mu, sigma, dof), the actual r, and the day.

    `df` must already be filtered to rows valid for this (horizon, feature
    set) -- no NaNs in the target or any sigma_col -- and sorted by time.
    """
    target = HORIZON_TARGET[horizon]
    r_all = df[target].to_numpy(dtype=float)
    Xs_all = df[sigma_cols].to_numpy(dtype=float)
    days = df["date"].to_numpy()
    cv = PurgedWalkForwardCV(n_splits=n_splits, embargo_frac=embargo_frac)
    let = label_end_times(df, horizon)
    ts = pd.to_datetime(df["time"]).reset_index(drop=True)

    rows = []
    for tr, te in cv.split(ts, label_end_times=let):
        mu_s, sd_s = _standardize_fit(Xs_all[tr])
        Xtr = (Xs_all[tr] - mu_s) / sd_s
        Xte = (Xs_all[te] - mu_s) / sd_s
        params = fit_location_scale(r_all[tr], Xtr)
        sig_te = predict_sigma(params, Xte)
        mu = params["mu"]
        dof = params["dof"]
        ll_g = gaussian_loglik(r_all[te], mu, sig_te)
        ll_t = student_t_loglik(r_all[te], mu, sig_te, dof)
        pit = gaussian_cdf(r_all[te], mu, sig_te)
        rows.append(pd.DataFrame({
            "day": days[te], "r": r_all[te], "mu": mu, "sigma": sig_te,
            "dof": dof, "ll_gauss": ll_g, "ll_t": ll_t, "pit": pit,
        }))
    out = pd.concat(rows, ignore_index=True)
    return {
        "per_row": out,
        "mean_ll_gauss": float(out["ll_gauss"].mean()),
        "mean_ll_t": float(out["ll_t"].mean()),
        "n": len(out),
    }


def unconditional_logscore(df: pd.DataFrame, horizon: str,
                           n_splits: int = N_SPLITS, embargo_frac: float = EMBARGO_FRAC) -> dict:
    """Baseline B(-1): constant Normal(train mean, train std) per fold --
    no conditioning at all. The floor every conditional model must beat."""
    target = HORIZON_TARGET[horizon]
    r_all = df[target].to_numpy(dtype=float)
    days = df["date"].to_numpy()
    cv = PurgedWalkForwardCV(n_splits=n_splits, embargo_frac=embargo_frac)
    let = label_end_times(df, horizon)
    ts = pd.to_datetime(df["time"]).reset_index(drop=True)
    rows = []
    for tr, te in cv.split(ts, label_end_times=let):
        mu = float(np.mean(r_all[tr]))
        sig = np.full(len(te), max(float(np.std(r_all[tr])), 1e-9))
        rows.append(pd.DataFrame({
            "day": days[te], "r": r_all[te],
            "ll_gauss": gaussian_loglik(r_all[te], mu, sig),
        }))
    out = pd.concat(rows, ignore_index=True)
    return {"per_row": out, "mean_ll_gauss": float(out["ll_gauss"].mean()), "n": len(out)}


# --------------------------------------------------------------------------
# incremental family test: (base + family) vs base, on a common sample
# --------------------------------------------------------------------------
def valid_mask(df: pd.DataFrame, horizon: str, cols: list) -> np.ndarray:
    m = df[HORIZON_TARGET[horizon]].notna().to_numpy().copy()
    for c in cols:
        m = m & df[c].notna().to_numpy()
    return m


def incremental_family_test(df: pd.DataFrame, horizon: str, base_cols: list,
                            family_cols: list, score: str = "ll_gauss",
                            n_boot: int = 2000, random_state: int = 7000) -> dict:
    """OOS log-score improvement of (base+family) over base on the common
    sample, with a day-clustered bootstrap CI on the mean per-row
    improvement."""
    full_cols = base_cols + [c for c in family_cols if c not in base_cols]
    m = valid_mask(df, horizon, full_cols)
    sub = df.loc[m].sort_values("time").reset_index(drop=True)
    base = cv_logscore(sub, horizon, base_cols)
    full = cv_logscore(sub, horizon, full_cols)
    b = base["per_row"][["day", score]].rename(columns={score: "base"})
    f = full["per_row"][[score]].rename(columns={score: "full"})
    joined = pd.concat([b, f], axis=1)
    joined["diff"] = joined["full"] - joined["base"]
    obs = float(joined["diff"].mean())

    # day-clustered bootstrap on the per-row improvement
    day_groups = {d: g["diff"].to_numpy() for d, g in joined.groupby("day")}
    day_keys = list(day_groups.keys())
    nd = len(day_keys)
    rng = np.random.default_rng(random_state)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, nd, size=nd)
        vals = np.concatenate([day_groups[day_keys[j]] for j in idx])
        boot[i] = vals.mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "horizon": horizon, "n": len(sub),
        "mean_ll_base": base[f"mean_{score}"] if f"mean_{score}" in base else base["mean_ll_gauss"],
        "mean_ll_full": full[f"mean_{score}"] if f"mean_{score}" in full else full["mean_ll_gauss"],
        "obs_improvement": obs, "boot_lo": float(lo), "boot_hi": float(hi),
        "base_result": base, "full_result": full,
    }


def day_swap_placebo(df: pd.DataFrame, horizon: str, base_cols: list,
                     family_cols: list, score: str = "ll_gauss",
                     n_perm: int = 1000, random_state: int = 7000,
                     observed: float | None = None) -> dict:
    """Day-swap null: permute the family columns' values across days
    (matched by within-day reading position), keep base + target tied to
    the real day, re-run the full purged-CV scale model, and record the
    log-score improvement. p = fraction of null improvements >= observed."""
    fam_only = [c for c in family_cols if c not in base_cols]
    full_cols = base_cols + fam_only
    m = valid_mask(df, horizon, full_cols)
    sub = df.loc[m].sort_values("time").reset_index(drop=True)
    if observed is None:
        base = cv_logscore(sub, horizon, base_cols)
        full = cv_logscore(sub, horizon, full_cols)
        observed = float(full["per_row"][score].mean() - base["per_row"][score].mean())

    base = cv_logscore(sub, horizon, base_cols)
    base_mean = float(base["per_row"][score].mean())

    days = sub["date"].to_numpy()
    uniq_days = pd.unique(days)
    day_rows = {d: np.nonzero(days == d)[0] for d in uniq_days}
    fam_vals = {c: sub[c].to_numpy(dtype=float) for c in fam_only}
    nd = len(uniq_days)
    rng = np.random.default_rng(random_state)

    null = np.empty(n_perm)
    sub_perm = sub.copy()
    for p in range(n_perm):
        donor = rng.permutation(nd)
        for c in fam_only:
            newv = np.empty(len(sub))
            for j, d in enumerate(uniq_days):
                recv = day_rows[d]
                give = day_rows[uniq_days[donor[j]]]
                if len(give) >= len(recv):
                    newv[recv] = fam_vals[c][give[: len(recv)]]
                else:
                    tmp = np.full(len(recv), np.nan)
                    tmp[: len(give)] = fam_vals[c][give]
                    # fill any remainder with the receiving day's own tail (rare)
                    tmp[np.isnan(tmp)] = fam_vals[c][recv][np.isnan(tmp)]
                    newv[recv] = tmp
            sub_perm[c] = newv
        full_p = cv_logscore(sub_perm, horizon, full_cols)
        null[p] = float(full_p["per_row"][score].mean()) - base_mean
    p_value = float(np.mean(null >= observed))
    return {"observed": observed, "null": null, "p_value": p_value, "n_perm": n_perm}


def day_swap_placebo_fast(df: pd.DataFrame, horizon: str, base_cols: list,
                          family_cols: list, n_perm: int = 2000,
                          random_state: int = 7000, observed: float | None = None,
                          ridge: float = 1e-8) -> dict:
    """Vectorized day-swap placebo for the Gaussian scale log-score.

    Same design as `day_swap_placebo` but hoists everything invariant across
    permutations out of the loop: fold indices, the fixed location mu and
    |r-mu| target (the target is NOT permuted, only the family features are),
    the fixed standardized base design, and the sigma floor. Only the family
    columns' train-standardization and one small normal-equations solve run
    per fold per permutation. Solver is normal equations (X'X is tiny) rather
    than SVD lstsq -- verified against the reference path in tests.
    """
    fam_only = [c for c in family_cols if c not in base_cols]
    full_cols = base_cols + fam_only
    m = valid_mask(df, horizon, full_cols)
    sub = df.loc[m].sort_values("time").reset_index(drop=True)

    r = sub[HORIZON_TARGET[horizon]].to_numpy(dtype=float)
    B = sub[base_cols].to_numpy(dtype=float)
    F = sub[fam_only].to_numpy(dtype=float)
    ts = pd.to_datetime(sub["time"]).reset_index(drop=True)
    let = label_end_times(sub, horizon)
    cv = PurgedWalkForwardCV(n_splits=N_SPLITS, embargo_frac=EMBARGO_FRAC)
    folds = list(cv.split(ts, label_end_times=let))

    # per-fold invariants
    F_pre = []
    base_mean_scores = []
    for tr, te in folds:
        mu = float(r[tr].mean())
        abs_tr = np.abs(r[tr] - mu)
        floor = SIGMA_FLOOR_FRAC * (float(np.std(r[tr])) or 1e-9)
        bmu, bsd = _standardize_fit(B[tr]); bsd = np.where(bsd < 1e-12, 1.0, bsd)
        Bt = (B[tr] - bmu) / bsd
        Be = (B[te] - bmu) / bsd
        # base-only score (fixed across perms)
        Xtr0 = np.column_stack([np.ones(len(tr)), Bt])
        Xte0 = np.column_stack([np.ones(len(te)), Be])
        beta0 = np.linalg.solve(Xtr0.T @ Xtr0 + ridge * np.eye(Xtr0.shape[1]), Xtr0.T @ abs_tr)
        sig0 = np.maximum(np.maximum(Xte0 @ beta0, 1e-12) * np.sqrt(np.pi / 2), floor)
        base_mean_scores.append(gaussian_loglik(r[te], mu, sig0))
        F_pre.append({"tr": tr, "te": te, "mu": mu, "abs_tr": abs_tr, "floor": floor,
                      "Bt": Bt, "Be": Be})
    base_mean = float(np.mean(np.concatenate(base_mean_scores)))

    if observed is None:
        inc = incremental_family_test(df, horizon, base_cols, family_cols, n_boot=1)
        observed = inc["obs_improvement"]

    # day structure for the swap
    days = sub["date"].to_numpy()
    uniq = pd.unique(days)
    day_rows = [np.nonzero(days == d)[0] for d in uniq]
    lens = np.array([len(x) for x in day_rows])
    nd = len(uniq)
    rng = np.random.default_rng(random_state)

    null = np.empty(n_perm)
    for p in range(n_perm):
        donor = rng.permutation(nd)
        Fp = np.empty_like(F)
        for j in range(nd):
            recv = day_rows[j]; give = day_rows[donor[j]]
            if lens[donor[j]] >= lens[j]:
                Fp[recv] = F[give[: lens[j]]]
            else:
                Fp[recv] = F[recv]  # donor too short: fall back to own (rare)
                Fp[recv[: lens[donor[j]]]] = F[give]
        scores = []
        for fp in F_pre:
            tr, te = fp["tr"], fp["te"]
            fmu = Fp[tr].mean(axis=0); fsd = Fp[tr].std(axis=0)
            fsd = np.where(fsd < 1e-12, 1.0, fsd)
            Ft = (Fp[tr] - fmu) / fsd
            Fe = (Fp[te] - fmu) / fsd
            Xtr = np.column_stack([np.ones(len(tr)), fp["Bt"], Ft])
            Xte = np.column_stack([np.ones(len(te)), fp["Be"], Fe])
            beta = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]), Xtr.T @ fp["abs_tr"])
            sig = np.maximum(np.maximum(Xte @ beta, 1e-12) * np.sqrt(np.pi / 2), fp["floor"])
            scores.append(gaussian_loglik(r[te], fp["mu"], sig))
        null[p] = float(np.mean(np.concatenate(scores))) - base_mean
    return {"observed": float(observed), "null": null,
            "p_value": float(np.mean(null >= observed)), "n_perm": n_perm}


# --------------------------------------------------------------------------
# calibration diagnostics
# --------------------------------------------------------------------------
def calibration_report(per_row: pd.DataFrame, taus_bps: list = [5, 10, 20]) -> dict:
    """PIT uniformity (KS stat) + P(|r|>tau) reliability, OOS."""
    from scipy.stats import kstest

    pit = per_row["pit"].to_numpy()
    pit = pit[np.isfinite(pit)]
    ks = float(kstest(pit, "uniform").statistic)
    rel = {}
    mu = per_row["mu"].to_numpy()
    sig = per_row["sigma"].to_numpy()
    r = per_row["r"].to_numpy()
    for tau_bps in taus_bps:
        tau = tau_bps / 1e4
        pred = prob_abs_gt(mu, sig, tau).mean()
        emp = float(np.mean(np.abs(r) > tau))
        rel[tau_bps] = {"predicted": float(pred), "empirical": emp}
    return {"pit_ks": ks, "exceedance": rel}
