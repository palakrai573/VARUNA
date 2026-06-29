"""
Reference forecasters the AI is benchmarked against.

  * persistence            : tomorrow = today (the classic hard-to-beat baseline)
  * persistence_of_anomaly : forecast = climatology(doy) + rho^k * anomaly(today)
  * climatology            : forecast = climatology(doy)  (no skill, the floor)

All operate in real units on the national grid, so they plug straight into the
same evaluation harness as the AI model.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

RHO = {"rain": 0.55, "tmax": 0.80, "tmin": 0.80}


def _doy(ts):
    return min(int(pd.Timestamp(ts).dayofyear), 365)


def climatology_forecast(carr, dates, t):
    frames = {v: np.empty((C.HORIZON, *carr[v].shape[1:]), dtype="float32") for v in C.VARIABLES}
    for k in range(C.HORIZON):
        d = _doy(dates[t] + pd.Timedelta(days=k))
        for v in C.VARIABLES:
            frames[v][k] = carr[v][d - 1]
    return frames


def persistence_of_anomaly(obs, carr, dates, t):
    """climatology(doy+k) + rho^k * (obs(t-1) - climatology(doy(t-1)))."""
    frames = {v: np.empty((C.HORIZON, *carr[v].shape[1:]), dtype="float32") for v in C.VARIABLES}
    d0 = _doy(dates[t - 1])
    for v in C.VARIABLES:
        anom0 = obs[v].values[t - 1] - carr[v][d0 - 1]
        rho = RHO[v]
        for k in range(C.HORIZON):
            d = _doy(dates[t] + pd.Timedelta(days=k))
            f = carr[v][d - 1] + (rho ** (k + 1)) * anom0
            frames[v][k] = np.clip(f, 0, None) if v == "rain" else f
    return frames


def persistence(obs, dates, t):
    """Repeat the last observed day for all leads."""
    frames = {}
    for v in C.VARIABLES:
        last = obs[v].values[t - 1]
        frames[v] = np.repeat(last[None], C.HORIZON, axis=0).astype("float32")
    return frames
