"""Touch (first-passage) probability engine — Q007 addendum (2026-07-27).

Companion to `stage1_density_engine.py`, NOT a replacement. That engine
estimates the conditional distribution of the TERMINAL return r_h at
exactly t+h. This module estimates a different, genuinely distinct
target: the probability that price moves at least tau AT ANY POINT
during (t, t+h] — the "touch" or first-passage event, which is what the
dashboard's primary probability cards actually need ("chance SPY moves
at least +/-10bps in the next 5 minutes").

Why not derive touch probability analytically from the terminal
distribution (e.g. the reflection-principle result P(touch)=2*P(terminal)
for a driftless continuous Brownian motion): Q007's own findings already
established that r_h is NOT Gaussian/diffusive at these horizons (fat
tails, skewed Hansen-t innovations, state-dependent scale fit
independently per horizon) — imposing a diffusion assumption just to get
a closed-form touch formula would be a WEAKER assumption than what's
already validated, not a stronger one. Since 1-minute high/low bars make
the touch event directly observable in history, this engine estimates it
empirically instead.

MODEL (mirrors stage1_density_engine.py's location-scale-shape structure,
substituting the terminal skew-t for a Weibull excursion model — the
standard extreme-value/running-maximum family):

    target: max_excursion_h = max_{0<s<=h} |r_s|   (see
    build_feature_table.compute_excursion_labels; TARGET, never a
    feature — nonnegative, no location term needed, unlike r_h itself.)

    scale_h(state):  OLS regression of max_excursion_h directly on the
        SAME six standardized state features as the terminal engine's
        sigma_h(state) (mtc, mtc_sqrt, mtc_sq, realized_vol_cum,
        realized_vol_30min, vix_close_lagged) -- predicts E[excursion_h |
        state], floored at 10% of the unconditional mean to prevent a
        degenerate near-zero scale.

    shape: standardized ratio w = max_excursion_h / scale_h(state) fit to
        a 2-parameter log-normal (shape sigma_h, scale lambda_h), constant
        per horizon -- same "constant shape, state-dependent scale" design
        as the terminal engine's constant skew-t (eta, lambda). Log-normal
        was chosen over the initially-tried Weibull by direct comparison:
        in-sample PIT KS 0.021 (log-normal) vs 0.133 (Weibull) vs 0.060
        (gamma) on the 5min horizon -- log-normal is also the standard
        family for volatility/magnitude quantities (multiplicative, not
        additive, dispersion), which matches a running-maximum-of-returns
        target better than an additive extreme-value family here.

    P_touch(tau | state) = P(max_excursion_h > tau | state)
                          = LogNormal_survival(tau / scale_h(state); s=sigma_h, scale=lambda_h)

Same hard constraint as the terminal engine (project direction
2026-07-21): regular trading hours only -- `predict()` returns a horizon
only if `now + h <= 16:00`. Reuses q007_skew.regular_hours_mask
(depends only on decision time + horizon, so it already guarantees no
after-hours bar ever enters this label the same way it does for the
terminal engine — see compute_excursion_labels' docstring).

Fit on development data only (frozen artifact convention matches
stage1_density_engine.py); a pooled refit script analogous to
refit_stage1_engine_pooled.py can be added the same way once this
engine is validated, following the same one-shot-holdout discipline.
"""

from __future__ import annotations

import datetime as _dt
import json
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm

import q007_distribution as q7
import q007_skew as sk
from stage1_density_engine import BASE_SCALE_COLS, _add_time_features

MOC = time(16, 0)
WINDOW_START = time(14, 0)
ENGINE_HORIZONS = ["5min", "10min", "15min", "20min", "30min"]
EXCURSION_TARGET = {h: f"max_excursion_{h}" for h in ENGINE_HORIZONS}
SCALE_FLOOR_FRAC = 0.10
ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "touch_probability_engine.json"


class TouchProbabilityEngine:
    """Frozen per-horizon touch-probability model: excursion scale(state)
    (OLS) + log-normal(s, scale) shape on the standardized ratio, constant
    per horizon."""

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    # ---- fit (development data only) ----
    def fit(self, dev: pd.DataFrame, horizons=ENGINE_HORIZONS) -> "TouchProbabilityEngine":
        t = pd.to_datetime(dev["time"])
        win = dev[(t.dt.time >= WINDOW_START) & (t.dt.time <= MOC)].copy()
        win = _add_time_features(win).sort_values("time").reset_index(drop=True)
        self.params = {"base_scale_cols": BASE_SCALE_COLS, "horizons": [], "by_horizon": {}}
        for h in horizons:
            target_col = EXCURSION_TARGET[h]
            m = sk.regular_hours_mask(win, h) & win[target_col].notna().to_numpy()
            for c in BASE_SCALE_COLS:
                m = m & win[c].notna().to_numpy()
            sub = win.loc[m]
            y = sub[target_col].to_numpy(float)
            X = sub[BASE_SCALE_COLS].to_numpy(float)
            xmu, xsd = q7._standardize_fit(X)
            xsd = np.where(xsd < 1e-12, 1.0, xsd)
            Xs = (X - xmu) / xsd
            coef = q7._ols_fit(Xs, y)
            pred_mean = np.maximum(q7._ols_predict(coef, Xs), 1e-12)
            floor = SCALE_FLOOR_FRAC * float(np.mean(y))
            scale = np.maximum(pred_mean, floor)
            w = y / scale
            s_hat, _loc, lam_hat = lognorm.fit(w, floc=0)
            self.params["horizons"].append(h)
            self.params["by_horizon"][h] = {
                "x_mu": xmu.tolist(), "x_sd": xsd.tolist(),
                "mean_excursion_coef": coef.tolist(),
                "floor": float(floor),
                "lognorm_s": float(s_hat), "lognorm_scale": float(lam_hat),
                "n_fit": int(len(sub)),
            }
        return self

    # ---- serialize / load ----
    def freeze(self, path: Path = ARTIFACT_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)
        return path

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH) -> "TouchProbabilityEngine":
        with open(path) as f:
            return cls(json.load(f))

    # ---- core computation on arrays ----
    def _scale(self, h: str, state: pd.DataFrame) -> np.ndarray:
        p = self.params["by_horizon"][h]
        X = state[BASE_SCALE_COLS].to_numpy(float)
        xmu = np.array(p["x_mu"]); xsd = np.array(p["x_sd"])
        coef = np.array(p["mean_excursion_coef"])
        Xs = (X - xmu) / xsd
        pred_mean = np.maximum(coef[0] + Xs @ coef[1:], 1e-12)
        return np.maximum(pred_mean, p["floor"])

    def touch_probs(self, h: str, state: pd.DataFrame, tau: float) -> dict:
        p = self.params["by_horizon"][h]
        scale = self._scale(h, state)
        w_tau = tau / scale
        p_touch = lognorm.sf(w_tau, p["lognorm_s"], loc=0, scale=p["lognorm_scale"])
        return {"scale": scale, "P_touch": p_touch, "lognorm_s": p["lognorm_s"], "lognorm_scale": p["lognorm_scale"]}

    # ---- real-time API: one decision point ----
    def predict(self, state: dict, now: time, tau_bps: float = 10.0) -> dict:
        """Mirrors Stage1DensityEngine.predict()'s signature/censoring,
        including returning {} before 14:00 -- `fit()` only trains on
        [14:00, 16:00] rows (see WINDOW_START above)."""
        if now < WINDOW_START:
            return {}
        row = dict(state)
        if "mtc" not in row and "minutes_to_close" in row:
            mtc = float(row["minutes_to_close"])
            row["mtc"] = mtc; row["mtc_sqrt"] = np.sqrt(max(mtc, 0)); row["mtc_sq"] = mtc ** 2
        sdf = pd.DataFrame([row])
        tau = tau_bps / 1e4
        out = {}
        for h in self.params["horizons"]:
            exit_t = (_dt.datetime.combine(_dt.date(2000, 1, 1), now)
                      + _dt.timedelta(minutes=q7.HORIZON_MINUTES[h])).time()
            if exit_t > MOC:
                continue
            tp = self.touch_probs(h, sdf, tau)
            out[h] = {"P_touch": float(tp["P_touch"][0]), "scale_bps": float(tp["scale"][0] * 1e4),
                      "lognorm_s": tp["lognorm_s"], "lognorm_scale": tp["lognorm_scale"]}
        return out


# --------------------------------------------------------------------------
# fit + validate + freeze (development only)
# --------------------------------------------------------------------------
def _oos_validation(dev: pd.DataFrame) -> dict:
    """OOS PIT KS (log-normal-CDF-transformed standardized excursion should
    be uniform if the model is calibrated) + mean log-normal log-likelihood,
    via purged walk-forward CV -- the same honest-deployment-read convention
    as stage1_density_engine.py's _oos_validation, refit per fold."""
    from scipy.stats import kstest

    t = pd.to_datetime(dev["time"])
    win = dev[(t.dt.time >= WINDOW_START) & (t.dt.time <= MOC)].copy()
    win = _add_time_features(win).sort_values("time").reset_index(drop=True)
    rep = {}
    for h in ENGINE_HORIZONS:
        target_col = EXCURSION_TARGET[h]
        m = sk.regular_hours_mask(win, h) & win[target_col].notna().to_numpy()
        for c in BASE_SCALE_COLS:
            m = m & win[c].notna().to_numpy()
        sub = win.loc[m].sort_values("time").reset_index(drop=True)
        y = sub[target_col].to_numpy(float)
        X = sub[BASE_SCALE_COLS].to_numpy(float)
        ts = pd.to_datetime(sub["time"]).reset_index(drop=True)
        let = q7.label_end_times(sub, h)
        cv = q7.PurgedWalkForwardCV(n_splits=q7.N_SPLITS, embargo_frac=q7.EMBARGO_FRAC)
        pit_all, ll_all = [], []
        for tr, te in cv.split(ts, label_end_times=let):
            xmu, xsd = q7._standardize_fit(X[tr]); xsd = np.where(xsd < 1e-12, 1.0, xsd)
            Xtr = (X[tr] - xmu) / xsd; Xte = (X[te] - xmu) / xsd
            coef = q7._ols_fit(Xtr, y[tr])
            pred_mean_tr = np.maximum(q7._ols_predict(coef, Xtr), 1e-12)
            floor = SCALE_FLOOR_FRAC * float(np.mean(y[tr]))
            scale_tr = np.maximum(pred_mean_tr, floor)
            w_tr = y[tr] / scale_tr
            s_hat, _loc, lam_hat = lognorm.fit(w_tr, floc=0)

            pred_mean_te = np.maximum(q7._ols_predict(coef, Xte), 1e-12)
            scale_te = np.maximum(pred_mean_te, floor)
            w_te = y[te] / scale_te
            pit_all.append(lognorm.cdf(w_te, s_hat, loc=0, scale=lam_hat))
            ll_all.append(lognorm.logpdf(w_te, s_hat, loc=0, scale=lam_hat) - np.log(scale_te))
        pit = np.concatenate(pit_all)
        ll = np.concatenate(ll_all)
        rep[h] = {"n": len(sub), "mean_logscore": float(np.mean(ll[np.isfinite(ll)])),
                  "pit_ks": float(kstest(np.clip(pit, 1e-6, 1 - 1e-6), "uniform").statistic)}
    return rep


def main():
    import data_access
    dev = data_access.load_development()
    eng = TouchProbabilityEngine().fit(dev)
    path = eng.freeze()
    print(f"[frozen Touch Probability Engine -> {path}]\n")

    print("Frozen per-horizon parameters:")
    for h in eng.params["horizons"]:
        p = eng.params["by_horizon"][h]
        print(f"  {h:>9}: lognorm_s={p['lognorm_s']:.3f}  lognorm_scale={p['lognorm_scale']:.4f}  n_fit={p['n_fit']}")

    print("\nOOS validation (purged walk-forward CV, development only):")
    rep = _oos_validation(dev)
    print(f"{'horizon':>9} | {'n':>6} {'mean_logscore':>14} {'PIT_KS':>8}")
    for h, r in rep.items():
        print(f"{h:>9} | {r['n']:>6} {r['mean_logscore']:>14.4f} {r['pit_ks']:>8.4f}")
    with open(Path(__file__).resolve().parent.parent / "logs" / "touch_probability_engine_validation.json", "w") as f:
        json.dump(rep, f, indent=2, default=float)

    print("\nDemo prediction @ 15:00 (mtc=60), median-ish vol state, tau=10bps:")
    demo = {"minutes_to_close": 60.0, "realized_vol_cum": 0.004,
            "realized_vol_30min": 0.0015, "vix_close_lagged": 16.0}
    pred = eng.predict(demo, now=time(15, 0), tau_bps=10.0)
    for h, d in pred.items():
        print(f"  {h:>9}: scale={d['scale_bps']:.1f}bps  P_touch={d['P_touch']:.3f}")
    print("\n[held-out test set NOT accessed — development data only]")


if __name__ == "__main__":
    main()
