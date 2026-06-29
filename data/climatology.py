"""
Smoothed day-of-year climatology.

The climatology is the heart of the anomaly forecasting approach: the model
learns deviations from the expected seasonal cycle, not the raw fields. This
keeps multi-day rollouts stable (they relax toward climatology instead of
diverging) and makes forecast skill meaningful.

    anomaly(t)     = obs(t) - climatology(dayofyear(t))
    reconstruct(t) = climatology(dayofyear(t)) + anomaly(t)
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def smooth_climatology(ds: xr.Dataset, var: str, window: int = 21) -> xr.Dataset:
    """Day-of-year mean with a circular rolling smooth.

    Returns a Dataset with dims (dayofyear, lat, lon), dayofyear in 1..366.
    """
    da = ds[var]
    doy = da["time"].dt.dayofyear
    clim = da.groupby(doy).mean("time", skipna=True)
    clim = clim.rename({"dayofyear": "dayofyear"}) if "dayofyear" in clim.dims else clim
    # Ensure a clean dayofyear coordinate.
    if "dayofyear" not in clim.dims:
        clim = clim.rename({list(clim.dims)[0]: "dayofyear"})

    # Circular smoothing across the year boundary.
    arr = clim.values  # (doy, lat, lon)
    n = arr.shape[0]
    pad = window // 2
    wrapped = np.concatenate([arr[-pad:], arr, arr[:pad]], axis=0)
    kernel = np.ones(window) / window
    smoothed = np.empty_like(arr)
    for i in range(n):
        seg = wrapped[i:i + window]                       # (window, lat, lon)
        smoothed[i] = np.nanmean(seg, axis=0)
    out = xr.DataArray(
        smoothed, dims=clim.dims,
        coords={"dayofyear": clim["dayofyear"].values,
                "lat": ds["lat"].values, "lon": ds["lon"].values},
        name=var,
    )
    return out.to_dataset()


def to_anomaly(field: np.ndarray, clim_var: xr.DataArray, doy: int) -> np.ndarray:
    """obs field -> anomaly using the climatology for `doy`."""
    doy = int(min(max(doy, 1), int(clim_var["dayofyear"].max())))
    return field - clim_var.sel(dayofyear=doy).values


def from_anomaly(anom: np.ndarray, clim_var: xr.DataArray, doy: int) -> np.ndarray:
    """anomaly -> reconstructed field using the climatology for `doy`."""
    doy = int(min(max(doy, 1), int(clim_var["dayofyear"].max())))
    return anom + clim_var.sel(dayofyear=doy).values
