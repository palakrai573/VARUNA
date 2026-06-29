"""
Digital-twin state update: observation-guided nudging + rolling bias correction.

Newtonian relaxation (the standard lightweight assimilation scheme):
    state = forecast + alpha * (observation - forecast)

Makes the "continuously evolving state fused from observations" claim concrete
and visually demonstrable (model-only vs model+obs), without overclaiming a full
4D-Var / EnKF system.
"""
from __future__ import annotations

import numpy as np


def nudge(forecast, observation, alpha=0.25):
    obs = np.where(np.isfinite(observation), observation, forecast)
    return forecast + alpha * (obs - forecast)


def bias_correct(forecast, recent_errors):
    if recent_errors is None or np.size(recent_errors) == 0:
        return forecast
    bias = np.nanmean(recent_errors, axis=0)
    return forecast - np.where(np.isfinite(bias), bias, 0.0)


def assimilate(forecast, observation, recent_errors=None, alpha=0.25, non_negative=False):
    state = bias_correct(forecast, recent_errors) if recent_errors is not None else forecast
    state = nudge(state, observation, alpha)
    return np.clip(state, 0, None) if non_negative else state


def innovation_stats(forecast, observation):
    d = observation - forecast
    m = np.isfinite(d)
    if m.sum() == 0:
        return {"mean": float("nan"), "rmse": float("nan")}
    return {"mean": float(np.mean(d[m])), "rmse": float(np.sqrt(np.mean(d[m] ** 2)))}


def optimal_interpolation(background, observation, length_scale=2.0,
                          sigma_b=1.0, sigma_o=0.6, landmask=None):
    """Optimal Interpolation (OI) analysis - a proper assimilation step.

        x_a = x_b + K (y - x_b),   K spreads the innovation per a spatially
        correlated background-error covariance B (modelled as a Gaussian of
        `length_scale` grid cells) with diagonal observation error R.

    With observations on the model grid (H = I), the gain reduces to a
    covariance-weighted, spatially-smoothed innovation:

        x_a = x_b + (sigma_b^2 / (sigma_b^2 + sigma_o^2)) * smooth_B(y - x_b)

    This spreads each observation's information to neighbours according to the
    correlation length - the defining behaviour of OI, well beyond pointwise
    nudging.
    """
    from scipy.ndimage import gaussian_filter

    bg = np.asarray(background, dtype="float64")
    d = np.where(np.isfinite(observation), observation - bg, 0.0)
    gain = sigma_b ** 2 / (sigma_b ** 2 + sigma_o ** 2)
    # smooth the innovation field with the background-error correlation kernel
    spread = gaussian_filter(d, sigma=length_scale, mode="nearest")
    analysis = bg + gain * spread
    if landmask is not None:
        analysis = np.where(landmask, analysis, bg)
    return analysis


def analysis_increment(background, analysis):
    """Diagnostic: the field the assimilation actually changed."""
    return analysis - background
