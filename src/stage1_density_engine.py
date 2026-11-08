"""Stage 1 Density Engine — Q007 deliverable (Phase 4, decision-engine Stage 1).

A frozen, calibrated estimator of the conditional forward-return distribution
for SPY between 14:00 and 16:00 EST, per horizon. Ingests market state knowable
at the decision timestamp and outputs, for each horizon whose exit is on/before
the 16:00 MOC cross:

    P_tail     = P(|r_h| > tau)          (opportunity / magnitude)
    P_upside   = P(r_h > +tau)           (directional up-tail)
    P_downside = P(r_h < -tau)           (directional down-tail)

plus the full predicted density parameters (mu, sigma, eta, lambda).

HARD CONSTRAINT (project direction 2026-07-21): regular trading hours only.
`predict()` returns a horizon only if `now + h <= 16:00` (no after-hours target
is ever produced). The Q000 after-hours reversal signal is out of scope.

MODEL (frozen from Q007 milestones 1-2, development data only):
- location mu_h: constant (the in-window conditional MEAN is a near-martingale,
  Q007 milestone 1: OOS R^2 ~ 0 at every horizon).
- scale sigma_h(state): milestone-1 abs-residual OLS on the vol-persistence +
  time-of-day baseline B1 = [minutes_to_close, sqrt, sq, realized_vol_cum,
  realized_vol_30min, vix_close_lagged]. This is the calibrated, OOS-validated
  driver of the distribution's width.
- shape: Hansen (1994) skewed-t with a CONSTANT skew lambda and dof eta.
  Milestone 2 found conditional skew (on imbalance/breadth/momentum) is NOT
  forecastable, but a constant unconditional skew is a small real calibration
  gain -- so lambda is a fixed per-horizon constant, and P_upside vs P_downside
  are near-symmetric BY FINDING, not by assumption.

The engine is fit on the FULL development set and frozen to JSON. The held-out
test set is never touched here.
"""

from __future__ import annotations

import datetime as _dt
import json
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

import q007_distribution as q7
import q007_skew as sk

MOC = time(16, 0)
WINDOW_START = time(14, 0)
BASE_SCALE_COLS = ["mtc", "mtc_sqrt", "mtc_sq",
                   "realized_vol_cum", "realized_vol_30min", "vix_close_lagged"]
ENGINE_HORIZONS = ["5min", "10min", "15min", "20min", "30min", "to_close"]
DEFAULT_TAUS_BPS = [5, 10, 20]
ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "stage1_density_engine.json"


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mtc = out["minutes_to_close"].astype(float)
    out["mtc"] = mtc
    out["mtc_sqrt"] = np.sqrt(np.clip(mtc, 0, None))
    out["mtc_sq"] = mtc ** 2
    return out


class Stage1DensityEngine:
    """Frozen per-horizon conditional density: mu (const) + sigma(state) +
    Hansen skewed-t(eta, const lambda)."""

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    # ---- fit (development data only) ----
    def fit(self, dev: pd.DataFrame, horizons=ENGINE_HORIZONS) -> "Stage1DensityEngine":
        t = pd.to_datetime(dev["time"])
        win = dev[(t.dt.time >= WINDOW_START) & (t.dt.time <= MOC)].copy()
        win = _add_time_features(win).sort_values("time").reset_index(drop=True)
        self.params = {"base_scale_cols": BASE_SCALE_COLS, "horizons": [], "by_horizon": {}}
        for h in horizons:
            m = sk.regular_hours_mask(win, h) & win[q7.HORIZON_TARGET[h]].notna().to_numpy()
            for c in BASE_SCALE_COLS:
                m = m & win[c].notna().to_numpy()
            sub = win.loc[m]
            r = sub[q7.HORIZON_TARGET[h]].to_numpy(float)
            X = sub[BASE_SCALE_COLS].to_numpy(float)
            # scale (milestone 1)
            xmu, xsd = q7._standardize_fit(X); xsd = np.where(xsd < 1e-12, 1.0, xsd)
            ps = q7.fit_location_scale(r, (X - xmu) / xsd)
            sig = q7.predict_sigma(ps, (X - xmu) / xsd)
            z = (r - ps["mu"]) / sig
            # shape (milestone 2): constant skew-t
            fit = sk.fit_skewt(z, const_skew=True)
            self.params["horizons"].append(h)
            self.params["by_horizon"][h] = {
                "x_mu": xmu.tolist(), "x_sd": xsd.tolist(),
                "absresid_coef": ps["coef"].tolist(), "loc_mu": ps["mu"],
                "sigma_floor": ps["floor"],
                "skew_m": fit["m"], "skew_s": fit["s"], "skew_eta": fit["eta"],
                "skew_lambda": float(np.tanh(fit["theta"][0])),
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
    def load(cls, path: Path = ARTIFACT_PATH) -> "Stage1DensityEngine":
        with open(path) as f:
            return cls(json.load(f))

    # ---- core density for one horizon on arrays ----
    def _sigma_mu(self, h: str, state: pd.DataFrame):
        p = self.params["by_horizon"][h]
        X = state[BASE_SCALE_COLS].to_numpy(float)
        xmu = np.array(p["x_mu"]); xsd = np.array(p["x_sd"])
        coef = np.array(p["absresid_coef"])
        Xs = (X - xmu) / xsd
        pred_abs = np.maximum(coef[0] + Xs @ coef[1:], 1e-12)
        sigma = np.maximum(pred_abs * np.sqrt(np.pi / 2), p["sigma_floor"])
        return sigma, p["loc_mu"], p

    def _fit_dict(self, p):
        return {"m": p["skew_m"], "s": p["skew_s"], "eta": p["skew_eta"],
                "theta": np.array([np.arctanh(np.clip(p["skew_lambda"], -0.999, 0.999))]),
                "symmetric": False, "const_skew": True}

    def tail_probs(self, h: str, state: pd.DataFrame, tau: float) -> dict:
        sigma, mu, p = self._sigma_mu(h, state)
        fd = self._fit_dict(p)
        p_up = sk.prob_r_gt(fd, tau, sigma, mu, None)
        p_dn = sk.prob_r_lt(fd, -tau, sigma, mu, None)
        return {"sigma": sigma, "mu": mu, "eta": p["skew_eta"], "lambda": p["skew_lambda"],
                "P_tail": p_up + p_dn, "P_upside": p_up, "P_downside": p_dn}

    # ---- real-time API: one decision point ----
    def predict(self, state: dict, now: time, tau_bps: float = 10.0) -> dict:
        """state: dict with the base scale features (or raw fields to derive
        them). now: decision time-of-day. Returns per-horizon densities for
        horizons whose exit is on/before 16:00.

        Returns {} for `now` before 14:00: `fit()` only trains on
        [14:00, 16:00] rows (see WINDOW_START above), so a prediction
        before then would extrapolate outside the window the scale/shape
        parameters were ever validated on -- the same discipline as the
        16:00 MOC censoring below, just on the other edge of the window."""
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
            if h != "to_close":
                exit_t = (_dt.datetime.combine(_dt.date(2000, 1, 1), now)
                          + _dt.timedelta(minutes=q7.HORIZON_MINUTES[h])).time()
                if exit_t > MOC:
                    continue
            tp = self.tail_probs(h, sdf, tau)
            out[h] = {"P_tail": float(tp["P_tail"][0]), "P_upside": float(tp["P_upside"][0]),
                      "P_downside": float(tp["P_downside"][0]), "sigma_bps": float(tp["sigma"][0] * 1e4),
                      "eta": tp["eta"], "lambda": tp["lambda"]}
        return out


# --------------------------------------------------------------------------
# fit + validate + freeze (development only)
# --------------------------------------------------------------------------
def _oos_validation(dev: pd.DataFrame) -> dict:
    """OOS PIT KS + log-score improvement (const-skew vs symmetric) per horizon,
    via purged walk-forward CV -- the honest deployment read for the frozen
    engine (fit is refit per fold with the same recipe)."""
    from scipy.stats import kstest
    t = pd.to_datetime(dev["time"])
    win = dev[(t.dt.time >= WINDOW_START) & (t.dt.time <= MOC)].copy()
    win = _add_time_features(win).sort_values("time").reset_index(drop=True)
    rep = {}
    for h in ENGINE_HORIZONS:
        m = sk.regular_hours_mask(win, h) & win[q7.HORIZON_TARGET[h]].notna().to_numpy()
        for c in BASE_SCALE_COLS:
            m = m & win[c].notna().to_numpy()
        sub = win.loc[m].sort_values("time").reset_index(drop=True)
        r = sub[q7.HORIZON_TARGET[h]].to_numpy(float); X = sub[BASE_SCALE_COLS].to_numpy(float)
        ts = pd.to_datetime(sub["time"]).reset_index(drop=True); let = q7.label_end_times(sub, h)
        cv = q7.PurgedWalkForwardCV(n_splits=q7.N_SPLITS, embargo_frac=q7.EMBARGO_FRAC)
        ls_c, ls_s, pit_c = [], [], []
        for tr, te in cv.split(ts, label_end_times=let):
            xmu, xsd = q7._standardize_fit(X[tr]); xsd = np.where(xsd < 1e-12, 1.0, xsd)
            ps = q7.fit_location_scale(r[tr], (X[tr] - xmu) / xsd)
            sig_te = q7.predict_sigma(ps, (X[te] - xmu) / xsd)
            z_tr = (r[tr] - ps["mu"]) / q7.predict_sigma(ps, (X[tr] - xmu) / xsd)
            fc = sk.fit_skewt(z_tr, const_skew=True); fs = sk.fit_skewt(z_tr, symmetric=True)
            ls_c.append(sk.skewt_logscore_r(fc, r[te], sig_te, ps["mu"], None))
            ls_s.append(sk.skewt_logscore_r(fs, r[te], sig_te, ps["mu"], None))
            pit_c.append(sk.skewt_cdf_r(fc, r[te], sig_te, ps["mu"], None))
        ls_c = np.concatenate(ls_c); ls_s = np.concatenate(ls_s); pit = np.concatenate(pit_c)
        rep[h] = {"n": len(sub), "mean_logscore": float(ls_c.mean()),
                  "logscore_gain_vs_symmetric": float(ls_c.mean() - ls_s.mean()),
                  "pit_ks": float(kstest(np.clip(pit, 1e-6, 1 - 1e-6), "uniform").statistic)}
    return rep


def main():
    import data_access
    dev = data_access.load_development()
    eng = Stage1DensityEngine().fit(dev)
    path = eng.freeze()
    print(f"[frozen Stage 1 Density Engine -> {path}]\n")

    print("Frozen per-horizon parameters:")
    for h in eng.params["horizons"]:
        p = eng.params["by_horizon"][h]
        print(f"  {h:>9}: eta={p['skew_eta']:.2f}  lambda={p['skew_lambda']:+.4f}  n_fit={p['n_fit']}")

    print("\nOOS validation (purged walk-forward CV, development only):")
    rep = _oos_validation(dev)
    print(f"{'horizon':>9} | {'n':>6} {'mean_logscore':>14} {'gain_vs_symm':>13} {'PIT_KS':>8}")
    for h, r in rep.items():
        print(f"{h:>9} | {r['n']:>6} {r['mean_logscore']:>14.4f} {r['logscore_gain_vs_symmetric']:>+13.5f} {r['pit_ks']:>8.4f}")
    with open(Path(__file__).resolve().parent.parent / "logs" / "stage1_density_engine_validation.json", "w") as f:
        json.dump(rep, f, indent=2, default=float)

    # demo prediction at a representative mid-window state
    print("\nDemo prediction @ 15:00 (mtc=60), median-ish vol state, tau=10bps:")
    demo = {"minutes_to_close": 60.0, "realized_vol_cum": 0.004,
            "realized_vol_30min": 0.0015, "vix_close_lagged": 16.0}
    pred = eng.predict(demo, now=time(15, 0), tau_bps=10.0)
    for h, d in pred.items():
        print(f"  {h:>9}: sigma={d['sigma_bps']:.1f}bps  P_tail={d['P_tail']:.3f}  "
              f"P_up={d['P_upside']:.3f}  P_dn={d['P_downside']:.3f}  (lambda={d['lambda']:+.3f})")
    print("\n[held-out test set NOT accessed — development data only]")


if __name__ == "__main__":
    main()
