"""Pure density-curve helpers for the live probability dashboard (`app.py`).

Renders the frozen Stage 1 Density Engine's predicted Hansen skew-t
distribution of forward returns as an (x_bps, pdf) curve, plus the tail
masks used to shade P_upside/P_downside beyond +/-tau. Reuses
`q007_skew.skewt_logpdf_std` directly -- no re-derivation of the density,
so a bug fixed there is fixed here too.

No Streamlit/plotly import here -- kept plot-library-agnostic and unit
testable; `app.py` turns this into a `plotly` figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q007_skew as sk  # noqa: E402


def skewt_pdf_bps(x_bps: np.ndarray, mu_bps: float, sigma_bps: float, eta: float, lam: float) -> np.ndarray:
    """Density of the forward return (in bps) under the Hansen skew-t with
    location `mu_bps` (~0, frozen constant), scale `sigma_bps` (live,
    state-dependent), shape (eta, lam) (frozen constants)."""
    z = (np.asarray(x_bps, dtype=float) - mu_bps) / sigma_bps
    logpdf = sk.skewt_logpdf_std(z, eta, lam) - np.log(sigma_bps)
    return np.exp(logpdf)


def density_curve(mu_bps: float, sigma_bps: float, eta: float, lam: float,
                  tau_bps: float = 10.0, span_sigma: float = 6.0, n: int = 400) -> dict:
    """(x_bps, pdf) over a +/-`span_sigma`*sigma grid, plus boolean masks for
    the upside/downside tail regions beyond +/-tau_bps -- everything
    `app.py` needs to draw the shaded density plot."""
    half = span_sigma * sigma_bps
    x = np.linspace(mu_bps - half, mu_bps + half, n)
    pdf = skewt_pdf_bps(x, mu_bps, sigma_bps, eta, lam)
    return {"x_bps": x, "pdf": pdf, "upside_mask": x >= tau_bps, "downside_mask": x <= -tau_bps}
