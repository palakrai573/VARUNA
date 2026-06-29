"""
Build the processed cache the model and dashboard consume.

Pipeline (REAL IMD data only):
  1. Download rain (0.25 deg) + tmax/tmin (1.0 deg) for all configured years.
  2. Regrid temperature onto the rainfall grid -> one common 0.25 deg national grid.
  3. Build a smoothed day-of-year climatology from TRAIN years only (no leakage).
  4. Compute anomalies and per-variable anomaly std (train years) for scaling.
  5. Save: obs.nc, clim_rain.nc, clim_temp.nc, stats.json, landmask.npy, meta.json.

Outputs land in data/processed/.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from data import download_imd, climatology  # noqa: E402


def _p(name):
    return os.path.join(C.PROCESSED_DIR, name)


def regrid_to_rain(temp_da: xr.DataArray, rain_lat, rain_lon) -> xr.DataArray:
    """Interpolate a 1 deg temperature field onto the 0.25 deg rain grid.

    Linear interp inside coverage; edges/holes filled by nearest valid value so
    the array stays dense (ocean cells are masked later by the land mask).
    """
    # linear interp with extrapolation beyond the 1 deg coverage (fill_value=None
    # tells scipy's interpolator to extrapolate rather than emit NaN at edges)
    out = temp_da.interp(lat=rain_lat, lon=rain_lon, method="linear",
                         kwargs={"fill_value": None})
    # any residual NaN (e.g. all-NaN day) -> that day's spatial mean
    out = out.fillna(out.mean(dim=["lat", "lon"]))
    return out


def build():
    print("[prepare] downloading IMD years:", C.ALL_YEARS, flush=True)
    dss = download_imd.download_years(C.ALL_YEARS)
    rain = dss["rain"]["rain"]                       # (time, 129, 135)
    rain_lat = rain["lat"].values
    rain_lon = rain["lon"].values

    print("[prepare] regridding temperature -> rain grid ...", flush=True)
    tmax = regrid_to_rain(dss["tmax"]["tmax"], rain_lat, rain_lon)
    tmin = regrid_to_rain(dss["tmin"]["tmin"], rain_lat, rain_lon)

    # Align all three on a shared daily time axis (intersection of dates).
    obs = xr.Dataset({"rain": rain, "tmax": tmax, "tmin": tmin})
    obs = obs.sortby("time")
    # drop duplicate times if any
    _, idx = np.unique(obs["time"].values, return_index=True)
    obs = obs.isel(time=np.sort(idx))

    # Land mask: cells where rainfall is valid on most days (IMD is NaN over ocean).
    valid_frac = np.isfinite(obs["rain"]).mean("time").values
    landmask = valid_frac > 0.5
    print(f"[prepare] land cells: {int(landmask.sum())}/{landmask.size}", flush=True)

    # Fill remaining NaNs so model inputs are dense (ocean handled by mask/loss).
    for v in C.VARIABLES:
        obs[v] = obs[v].fillna(0.0 if v == "rain" else obs[v].mean())

    # --- climatology from TRAIN years only ---
    train_mask = obs["time"].dt.year.isin(C.CLIM_YEARS)
    obs_train = obs.sel(time=train_mask)
    print("[prepare] building smoothed climatology (train years) ...", flush=True)
    clim_rain = climatology.smooth_climatology(obs_train, "rain", C.CLIM_SMOOTH_WINDOW)
    clim_temp = xr.merge([
        climatology.smooth_climatology(obs_train, "tmax", C.CLIM_SMOOTH_WINDOW),
        climatology.smooth_climatology(obs_train, "tmin", C.CLIM_SMOOTH_WINDOW),
    ])
    clim = {"rain": clim_rain["rain"], "tmax": clim_temp["tmax"], "tmin": clim_temp["tmin"]}

    # --- anomaly statistics (std) on train years, for scaling model targets ---
    print("[prepare] computing anomaly std (train years) ...", flush=True)
    doy_all = obs["time"].dt.dayofyear.values
    stats = {}
    for v in C.VARIABLES:
        cv = clim[v]
        # anomaly only over train years, land cells
        tr_times = obs_train["time"].dt.dayofyear.values
        anom_tr = np.empty((obs_train.sizes["time"], *landmask.shape), dtype="float32")
        ov = obs_train[v].values
        for i, d in enumerate(tr_times):
            anom_tr[i] = ov[i] - cv.sel(dayofyear=int(min(d, int(cv["dayofyear"].max())))).values
        a = anom_tr[:, landmask]
        stats[v] = {
            "anom_std": float(np.nanstd(a)),
            "anom_mean": float(np.nanmean(a)),
            "clim_max": float(np.nanmax(cv.values)),
        }
        print(f"   {v}: anom_std={stats[v]['anom_std']:.4f}", flush=True)

    # --- save ---
    enc = lambda vs: {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in vs}
    obs.to_netcdf(_p("obs.nc"), encoding=enc(C.VARIABLES))
    clim_rain.to_netcdf(_p("clim_rain.nc"), encoding=enc(["rain"]))
    clim_temp.to_netcdf(_p("clim_temp.nc"), encoding=enc(["tmax", "tmin"]))
    np.save(_p("landmask.npy"), landmask)
    np.savez(_p("grid.npz"), lat=rain_lat, lon=rain_lon)
    with open(_p("stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    meta = {
        "source": "IMD (imdlib)", "train_years": C.TRAIN_YEARS,
        "val_years": C.VAL_YEARS, "test_years": C.TEST_YEARS,
        "n_time": int(obs.sizes["time"]), "grid": [int(landmask.shape[0]), int(landmask.shape[1])],
        "first": str(obs["time"].values[0])[:10], "last": str(obs["time"].values[-1])[:10],
    }
    with open(_p("meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare] done. obs={dict(obs.sizes)}  {meta['first']}..{meta['last']}")


if __name__ == "__main__":
    build()
