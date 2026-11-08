"""Tests for q007_skew.py (Hansen skewed-t)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

import q007_skew as sk
import q007_distribution as q7


def test_std_skewt_density_integrates_to_one():
    for eta, lam in [(6.0, 0.0), (8.0, 0.4), (5.0, -0.3)]:
        area, _ = quad(lambda z: np.exp(sk.skewt_logpdf_std(z, eta, lam)), -40, 40)
        assert abs(area - 1.0) < 1e-4


def test_std_skewt_mean_zero_var_one():
    eta, lam = 7.0, 0.5
    m0, _ = quad(lambda z: z * np.exp(sk.skewt_logpdf_std(z, eta, lam)), -40, 40)
    m2, _ = quad(lambda z: z * z * np.exp(sk.skewt_logpdf_std(z, eta, lam)), -40, 40)
    assert abs(m0) < 1e-3          # mean 0 by construction
    assert abs(m2 - 1.0) < 1e-3    # var 1 by construction


def test_cdf_matches_integral_of_pdf():
    eta, lam = 6.0, 0.35
    for z in [-2.0, -0.5, 0.7, 2.5]:
        num, _ = quad(lambda t: np.exp(sk.skewt_logpdf_std(t, eta, lam)), -40, z)
        assert abs(sk.skewt_cdf_std(z, eta, lam) - num) < 1e-4


def test_symmetric_reduces_to_standardized_t():
    # lambda=0 standardized Hansen-t == unit-variance Student-t
    from scipy.stats import t as tdist
    eta = 8.0
    zs = np.array([-1.5, 0.0, 1.2])
    got = np.exp(sk.skewt_logpdf_std(zs, eta, 0.0))
    scale = np.sqrt((eta - 2) / eta)  # unit-variance t
    exp = tdist.pdf(zs / scale, eta) / scale
    assert np.allclose(got, exp, atol=1e-9)


def test_mle_recovers_injected_skew():
    rng = np.random.default_rng(0)
    # sample from a known standardized skew-t via inverse-CDF on a grid
    eta_true, lam_true = 7.0, 0.5
    grid = np.linspace(-30, 30, 60001)
    cdf = sk.skewt_cdf_std(grid, eta_true, lam_true)
    u = rng.uniform(size=40000)
    z = np.interp(u, cdf, grid)
    fit = sk.fit_skewt(z, symmetric=False, const_skew=True)
    assert abs(fit["m"]) < 0.05
    assert abs(fit["s"] - 1.0) < 0.06
    assert abs(np.tanh(fit["theta"][0]) - lam_true) < 0.08
    assert abs(fit["eta"] - eta_true) < 2.0


def test_conditional_skew_detected_and_null_when_absent():
    rng = np.random.default_rng(1)
    n = 30000
    S = rng.normal(size=(n, 1))
    grid = np.linspace(-30, 30, 40001)
    # lambda depends on S: lam_i = tanh(1.2 * S_i)
    z = np.empty(n)
    u = rng.uniform(size=n)
    for lamv in np.unique(np.round(np.tanh(1.2 * S[:, 0]), 2)):
        pass  # (vectorized below instead)
    lam_i = np.tanh(1.2 * S[:, 0])
    # sample per-row by inverse cdf (batch by binning lambda for speed)
    order = np.argsort(lam_i)
    zt = np.empty(n)
    for idx in np.array_split(order, 40):
        lv = float(np.mean(lam_i[idx]))
        cdf = sk.skewt_cdf_std(grid, 7.0, lv)
        zt[idx] = np.interp(u[idx], cdf, grid)
    z = zt
    fit_c = sk.fit_skewt(z, S=S, symmetric=False, const_skew=False)
    # the S-coefficient should be clearly positive
    assert fit_c["theta"][1] > 0.3
    # with random S unrelated to z, coefficient ~ 0
    S_rand = rng.normal(size=(n, 1))
    fit_n = sk.fit_skewt(z, S=S_rand, symmetric=False, const_skew=False)
    assert abs(fit_n["theta"][1]) < 0.15


def test_regular_hours_mask_censors_after_1600():
    import pandas as pd
    times = pd.to_datetime(["2023-01-03 15:45", "2023-01-03 15:50",
                            "2023-01-03 15:55", "2023-01-03 16:00"])
    df = pd.DataFrame({"time": times, "date": pd.Timestamp("2023-01-03")})
    m15 = sk.regular_hours_mask(df, "15min")
    m10 = sk.regular_hours_mask(df, "10min")
    assert list(m15) == [True, False, False, False]   # only <=15:45 exits by 16:00
    assert list(m10) == [True, True, False, False]     # <=15:50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
