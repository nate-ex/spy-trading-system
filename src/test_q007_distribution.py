"""Tests for q007_distribution.py — the conditional location-scale harness.

Synthetic data only. Verifies (1) the log-likelihood math matches scipy,
(2) a genuine heteroskedastic driver improves the OOS log-score and clears
the day-swap placebo, (3) a pure-noise feature does not, and (4) the model
is calibrated (PIT ~ uniform, P(|r|>tau) predicted ~ empirical) when the
data-generating process matches the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as tdist

import q007_distribution as q7


def _synth(n_days=120, per_day=40, beta_scale=0.0, seed=0, noise_feat=False):
    """Build a synthetic 1-min-like panel with a heteroskedastic driver.

    return_5min ~ Normal(0, sigma) with log-sigma linear in `driver` when
    beta_scale>0 (a real scale signal). Times are 14:00 + minute index so
    label_end_times / purging behave like the real window.
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_day = pd.Timestamp("2023-01-02")
    d = base_day
    made = 0
    while made < n_days:
        if d.weekday() < 5:
            t0 = d + pd.Timedelta(hours=14)
            driver = rng.normal(size=per_day)
            sigma = 0.001 * np.exp(beta_scale * driver)
            r = rng.normal(0.0, sigma)
            noise = rng.normal(size=per_day)
            rows.append(pd.DataFrame({
                "time": [t0 + pd.Timedelta(minutes=i) for i in range(per_day)],
                "date": d.normalize(),
                "driver": driver,
                "noise": noise,
                "minutes_to_close": np.linspace(120, 5, per_day),
                "return_5min": r,
                "return_10min": r, "return_15min": r, "return_30min": r,
                "return_to_close": r,
            }))
            made += 1
        d += pd.Timedelta(days=1)
    return pd.concat(rows, ignore_index=True)


def test_gaussian_loglik_matches_scipy():
    r = np.array([0.001, -0.002, 0.0005])
    sig = np.array([0.001, 0.002, 0.0015])
    got = q7.gaussian_loglik(r, 0.0, sig)
    exp = norm.logpdf(r, 0.0, sig)
    assert np.allclose(got, exp)


def test_student_t_loglik_matches_scipy_variance_matched():
    r = np.array([0.001, -0.002, 0.0005])
    sig = np.array([0.001, 0.002, 0.0015])
    dof = 6.0
    got = q7.student_t_loglik(r, 0.0, sig, dof)
    s = sig * np.sqrt((dof - 2) / dof)
    exp = tdist.logpdf((r) / s, dof) - np.log(s)
    assert np.allclose(got, exp, atol=1e-8)


def test_prob_abs_gt_matches_direct():
    mu, sig, tau = 0.0, np.array([0.001]), 0.002
    p = q7.prob_abs_gt(mu, sig, tau)[0]
    exp = norm.sf(2.0) + norm.cdf(-2.0)
    assert abs(p - exp) < 1e-9


def test_real_scale_signal_improves_logscore():
    df = _synth(beta_scale=0.8, seed=1)
    res = q7.incremental_family_test(df, "5min", base_cols=["minutes_to_close"],
                                     family_cols=["driver"], n_boot=500)
    # a genuine scale driver should raise the OOS log-score, CI above 0
    assert res["obs_improvement"] > 0
    assert res["boot_lo"] > 0


def test_noise_feature_does_not_improve():
    df = _synth(beta_scale=0.8, seed=2)
    res = q7.incremental_family_test(df, "5min", base_cols=["minutes_to_close", "driver"],
                                     family_cols=["noise"], n_boot=500)
    # adding pure noise on top of the true driver should not help; CI spans 0
    assert res["boot_lo"] < 0 < res["boot_hi"] or abs(res["obs_improvement"]) < 1e-3


def test_placebo_rejects_noise_and_passes_signal():
    df = _synth(beta_scale=0.8, seed=3)
    sig = q7.day_swap_placebo(df, "5min", base_cols=["minutes_to_close"],
                              family_cols=["driver"], n_perm=200, random_state=1)
    noise = q7.day_swap_placebo(df, "5min", base_cols=["minutes_to_close"],
                                family_cols=["noise"], n_perm=200, random_state=1)
    assert sig["p_value"] < 0.05      # real driver beats the day-swap null
    assert noise["p_value"] > 0.10    # noise does not


def test_calibration_on_well_specified_data():
    df = _synth(beta_scale=0.8, seed=4)
    res = q7.cv_logscore(df, "5min", ["minutes_to_close", "driver"])
    cal = q7.calibration_report(res["per_row"])
    # PIT should be roughly uniform when the model matches the DGP
    assert cal["pit_ks"] < 0.08
    for tau_bps, d in cal["exceedance"].items():
        assert abs(d["predicted"] - d["empirical"]) < 0.06


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
