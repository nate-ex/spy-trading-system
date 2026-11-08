"""Unit tests for the density-curve helpers used by the live dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dashboard_density as dd  # noqa: E402


def test_skewt_pdf_bps_symmetric_case_peaks_near_mu():
    x = np.linspace(-50, 50, 1001)
    pdf = dd.skewt_pdf_bps(x, mu_bps=0.0, sigma_bps=10.0, eta=6.0, lam=0.0)
    assert x[np.argmax(pdf)] == 0.0


def test_skewt_pdf_bps_integrates_to_one():
    x = np.linspace(-200, 200, 20001)
    pdf = dd.skewt_pdf_bps(x, mu_bps=0.0, sigma_bps=10.0, eta=6.0, lam=-0.05)
    area = np.trapezoid(pdf, x)
    assert abs(area - 1.0) < 1e-3


def test_skewt_pdf_bps_wider_sigma_lowers_peak():
    x = np.array([0.0])
    peak_narrow = dd.skewt_pdf_bps(x, 0.0, 5.0, 6.0, 0.0)[0]
    peak_wide = dd.skewt_pdf_bps(x, 0.0, 20.0, 6.0, 0.0)[0]
    assert peak_wide < peak_narrow


def test_density_curve_masks_beyond_tau():
    curve = dd.density_curve(mu_bps=0.0, sigma_bps=10.0, eta=6.0, lam=0.0, tau_bps=10.0, n=201)
    assert curve["upside_mask"].sum() > 0
    assert curve["downside_mask"].sum() > 0
    assert not np.any(curve["upside_mask"] & curve["downside_mask"])
    assert (curve["x_bps"][curve["upside_mask"]] >= 10.0).all()
    assert (curve["x_bps"][curve["downside_mask"]] <= -10.0).all()


def test_density_curve_span_scales_with_sigma():
    narrow = dd.density_curve(0.0, 5.0, 6.0, 0.0, span_sigma=6.0, n=11)
    wide = dd.density_curve(0.0, 20.0, 6.0, 0.0, span_sigma=6.0, n=11)
    assert wide["x_bps"].max() > narrow["x_bps"].max()
