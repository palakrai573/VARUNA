"""
Download REAL IMD gridded data via imdlib. No synthetic data anywhere.

Datasets (https://www.imdpune.gov.in/cmpg/Griddata/):
  rain  -> Gridded Rainfall 0.25 x 0.25  (135 lon x 129 lat)
  tmax  -> Maximum Temperature 1.0 x 1.0 (31 x 31)
  tmin  -> Minimum Temperature 1.0 x 1.0 (31 x 31)

Offline-first: a year already cached as .grd under data/raw is read locally;
only missing years hit the IMD server. Fill/missing values are masked to NaN.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

# imdlib internal variable names map to its download labels.
_VARMAP = {"rain": "rain", "tmax": "tmax", "tmin": "tmin"}


def _clean(ds: xr.Dataset, var: str, lo: float, hi: float) -> xr.Dataset:
    """Mask IMD fill sentinels and physically impossible values to NaN."""
    da = ds[var]
    for f in C.IMD_FILL:
        da = da.where(da != f)
    da = da.where((da >= lo) & (da <= hi))
    return da.to_dataset(name=var)


def download_var(var: str, start_yr: int, end_yr: int) -> xr.Dataset:
    """Return a cleaned xr.Dataset for one variable across [start_yr, end_yr]."""
    import imdlib as imd  # heavy; lazy import

    try:
        data = imd.open_data(_VARMAP[var], start_yr, end_yr, "yearwise", C.RAW_DIR)
    except Exception:  # not cached -> fetch from IMD
        data = imd.get_data(_VARMAP[var], start_yr, end_yr,
                            fn_format="yearwise", file_dir=C.RAW_DIR)
    ds = data.get_xarray()
    name = var if var in ds.data_vars else list(ds.data_vars)[0]
    ds = ds.rename({name: var})
    if var == "rain":
        return _clean(ds, var, 0.0, 2000.0)
    return _clean(ds, var, -40.0, 60.0)


def download_years(years) -> dict:
    """Download every variable across `years`. Returns {var: concatenated ds}."""
    out = {}
    y0, y1 = min(years), max(years)
    for var in C.VARIABLES:
        print(f"[download] {var}  {y0}-{y1} ...", flush=True)
        out[var] = download_var(var, y0, y1)
        n = int(np.isfinite(out[var][var]).sum())
        if n == 0:
            raise RuntimeError(f"IMD {var} {y0}-{y1} returned no valid data.")
    return out


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:] if a.isdigit()] or C.ALL_YEARS
    dss = download_years(yrs)
    for v, ds in dss.items():
        print(f"  {v}: {dict(ds.sizes)}  mean={float(np.nanmean(ds[v])):.3f}")
