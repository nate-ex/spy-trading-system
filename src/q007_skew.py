"""Q007 milestone 2 — asymmetric shape via Hansen (1994) skewed Student-t.

Models the conditional forward-return distribution's SHAPE (skew + tails) on
top of milestone 1's validated conditional SCALE. Two-stage per fold:

  1. scale sigma_hat(state) from the milestone-1 abs-residual OLS
     (q007_distribution), giving standardized residual z = (r - mu)/sigma_hat;
  2. a Hansen skewed-t fit on z with free location m, scale s, dof eta, and
     skew lambda -- where lambda may be CONDITIONED on state:
     lambda_i = tanh(theta0 + theta . S_i), tanh keeping lambda in (-1, 1).

Everything is scored by an out-of-sample proper scoring rule (skew-t
log-likelihood of r) and PIT calibration via the skew-t CDF -- never R^2.

HARD CONSTRAINT (project direction 2026-07-21): regular trading hours only.
A horizon-h decision row is valid only if it exits on/before the 16:00 MOC
cross (t + h <= 16:00). `regular_hours_mask` enforces this; no post-16:00
(after-hours) price is ever used in a target.

Hansen standardized skew-t (mean 0, var 1), density of the standardized
variable T, parameters eta>2, lambda in (-1,1):
  c = Gamma((eta+1)/2) / (sqrt(pi (eta-2)) Gamma(eta/2))
  a = 4 lambda c (eta-2)/(eta-1);  b = sqrt(1 + 3 lambda^2 - a^2)
  f(z) = b c [1 + 1/(eta-2) ((b z + a)/(1 -/+ lambda))^2]^(-(eta+1)/2)
         (1-lambda if z < -a/b, else 1+lambda)
CDF via the standard Student-t CDF T_cdf(.;eta) (derivation in tests):
  x = (b z + a)/(1 -/+ lambda) * sqrt(eta/(eta-2))
  F(z) = (1-lambda) T_cdf(x)                       if z < -a/b
       = (1-lambda)/2 + (1+lambda)(T_cdf(x) - 1/2) if z >= -a/b

No real data embedded; generic.
"""

from __future__ import annotations

import datetime as _dt
from datetime import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as _tdist

import q007_distribution as q7

MOC = time(16, 0)


# --------------------------------------------------------------------------
# regular-hours censoring (hard constraint)
# --------------------------------------------------------------------------
def regular_hours_mask(df: pd.DataFrame, horizon: str) -> np.ndarray:
    """True where a horizon-h decision exits on/before the 16:00 MOC cross.
    to_close exits exactly at 16:00 (always valid intraday)."""
    tod = pd.to_datetime(df["time"]).dt.time.to_numpy()
    if horizon == "to_close":
        return tod < np.array(MOC, dtype=object)  # any pre-close bar
    hmin = q7.HORIZON_MINUTES[horizon]
    cutoff = (_dt.datetime.combine(_dt.date(2000, 1, 1), MOC) - _dt.timedelta(minutes=hmin)).time()
    return tod <= np.array(cutoff, dtype=object)


# --------------------------------------------------------------------------
# Hansen skewed-t: standardized density / cdf
# --------------------------------------------------------------------------
def _abc(eta: np.ndarray, lam: np.ndarray):
    logc = gammaln((eta + 1) / 2) - 0.5 * np.log(np.pi * (eta - 2)) - gammaln(eta / 2)
    c = np.exp(logc)
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(np.maximum(1 + 3 * lam ** 2 - a ** 2, 1e-12))
    return a, b, c, logc


def skewt_logpdf_std(z, eta, lam):
    """log density of the standardized Hansen skew-t (mean0 var1). eta, lam
    may be scalars or arrays broadcastable to z."""
    z = np.asarray(z, dtype=float)
    eta = np.asarray(eta, dtype=float); lam = np.asarray(lam, dtype=float)
    a, b, c, logc = _abc(eta, lam)
    denom = np.where(z < -a / b, 1 - lam, 1 + lam)
    arg = (b * z + a) / denom
    return np.log(b) + logc - (eta + 1) / 2 * np.log1p(arg ** 2 / (eta - 2))


def skewt_cdf_std(z, eta, lam):
    z = np.asarray(z, dtype=float)
    eta = np.asarray(eta, dtype=float); lam = np.asarray(lam, dtype=float)
    a, b, _, _ = _abc(eta, lam)
    scale = np.sqrt(eta / (eta - 2))
    lower = z < -a / b
    denom = np.where(lower, 1 - lam, 1 + lam)
    x = (b * z + a) / denom * scale
    tcdf = _tdist.cdf(x, eta)
    return np.where(lower, (1 - lam) * tcdf, (1 - lam) / 2 + (1 + lam) * (tcdf - 0.5))


# --------------------------------------------------------------------------
# MLE fit of a (possibly conditional-skew) skew-t on standardized residuals z
# --------------------------------------------------------------------------
def _lambda(theta, S):
    if S is None or len(theta) <= 1:
        base = theta[0] if len(theta) >= 1 else 0.0
        return np.tanh(base)
    return np.tanh(theta[0] + S @ theta[1:])


def _neg_loglik(params, z, S, symmetric):
    m, ls, leta = params[0], params[1], params[2]
    s = np.exp(ls)
    eta = 2.05 + np.exp(leta)
    if symmetric:
        lam = 0.0
    else:
        theta = params[3:]
        lam = _lambda(theta, S)
    zz = (z - m) / s
    ll = skewt_logpdf_std(zz, eta, lam) - np.log(s)
    if not np.all(np.isfinite(ll)):
        return 1e12
    return -np.sum(ll)


def fit_skewt(z: np.ndarray, S: np.ndarray | None = None,
              symmetric: bool = False, const_skew: bool = False) -> dict:
    """Fit skew-t on standardized residuals z.
    - symmetric: lambda = 0.
    - const_skew: lambda = tanh(theta0) (ignores S).
    - else: lambda = tanh(theta0 + S.theta) (conditional).
    Returns dict with m, s, eta, theta, symmetric flag, and the S column count.
    """
    z = np.asarray(z, dtype=float)
    x0 = [float(np.mean(z)), float(np.log(np.std(z) + 1e-9)), np.log(4.0)]
    if symmetric:
        S_use = None
        p0 = x0
    elif const_skew or S is None:
        S_use = None
        p0 = x0 + [0.0]
    else:
        S_use = S
        p0 = x0 + [0.0] + [0.0] * S.shape[1]
    res = minimize(_neg_loglik, p0, args=(z, S_use, symmetric),
                   method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    p = res.x
    m, ls, leta = p[0], p[1], p[2]
    return {"m": float(m), "s": float(np.exp(ls)), "eta": float(2.05 + np.exp(leta)),
            "theta": np.array(p[3:]), "symmetric": symmetric,
            "const_skew": (const_skew or (S is None and not symmetric)),
            "nll": float(res.fun)}


def model_lambda(fit: dict, S: np.ndarray | None) -> np.ndarray:
    if fit["symmetric"]:
        return 0.0
    theta = fit["theta"]
    if fit["const_skew"] or S is None or len(theta) <= 1:
        return np.tanh(theta[0]) if len(theta) else 0.0
    return np.tanh(theta[0] + S @ theta[1:])


def skewt_logscore_r(fit: dict, r: np.ndarray, sigma_hat: np.ndarray,
                     mu: float, S: np.ndarray | None) -> np.ndarray:
    """Absolute log-density of r under the fitted two-stage model."""
    z = (r - mu) / sigma_hat
    zz = (z - fit["m"]) / fit["s"]
    lam = model_lambda(fit, S)
    return skewt_logpdf_std(zz, fit["eta"], lam) - np.log(fit["s"]) - np.log(sigma_hat)


def skewt_cdf_r(fit: dict, r: np.ndarray, sigma_hat: np.ndarray,
                mu: float, S: np.ndarray | None) -> np.ndarray:
    z = (r - mu) / sigma_hat
    zz = (z - fit["m"]) / fit["s"]
    lam = model_lambda(fit, S)
    return skewt_cdf_std(zz, fit["eta"], lam)


def prob_r_gt(fit: dict, tau: float, sigma_hat: np.ndarray, mu: float,
              S: np.ndarray | None) -> np.ndarray:
    """P(r > tau)."""
    return 1.0 - skewt_cdf_r(fit, np.full_like(sigma_hat, tau), sigma_hat, mu, S)


def prob_r_lt(fit: dict, tau: float, sigma_hat: np.ndarray, mu: float,
              S: np.ndarray | None) -> np.ndarray:
    """P(r < tau)."""
    return skewt_cdf_r(fit, np.full_like(sigma_hat, tau), sigma_hat, mu, S)
