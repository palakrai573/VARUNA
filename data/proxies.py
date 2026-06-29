"""
Spatial proxy fields for the connected-applications layer.

Until GHSL built-up / INSAT-AOD PM2.5 are wired in, urban fraction and baseline
PM2.5 are derived from major-city locations. Clearly labeled as proxies; the
scenario engine consumes them to demonstrate the urban-heat and AQI applications.
"""
from __future__ import annotations

import numpy as np

# (lat, lon, peak built-up 0..1, sigma in degrees) for major Indian cities.
_CITIES = {
    "Delhi": (28.61, 77.21, 0.95, 0.6), "Mumbai": (19.08, 72.88, 0.95, 0.5),
    "Kolkata": (22.57, 88.36, 0.9, 0.5), "Chennai": (13.08, 80.27, 0.85, 0.5),
    "Bengaluru": (12.97, 77.59, 0.85, 0.5), "Hyderabad": (17.39, 78.49, 0.8, 0.5),
    "Ahmedabad": (23.03, 72.58, 0.8, 0.5), "Pune": (18.52, 73.86, 0.7, 0.4),
    "Surat": (21.17, 72.83, 0.7, 0.4), "Jaipur": (26.91, 75.79, 0.65, 0.4),
    "Lucknow": (26.85, 80.95, 0.65, 0.4), "Kanpur": (26.45, 80.33, 0.6, 0.35),
    "Nagpur": (21.15, 79.09, 0.6, 0.35), "Kochi": (9.93, 76.27, 0.55, 0.35),
    "Coimbatore": (11.02, 76.96, 0.55, 0.35), "Patna": (25.59, 85.14, 0.6, 0.35),
    "Bhopal": (23.26, 77.41, 0.55, 0.35), "Visakhapatnam": (17.69, 83.22, 0.6, 0.35),
}


def urban_fraction(lats, lons):
    """Gaussian bumps at city centres -> built-up fraction (0..1)."""
    lon2d, lat2d = np.meshgrid(lons, lats)
    frac = np.zeros_like(lon2d, dtype="float64")
    for _, (la, lo, pk, sg) in _CITIES.items():
        frac += pk * np.exp(-(((lon2d - lo) / sg) ** 2 + ((lat2d - la) / sg) ** 2))
    return np.clip(frac + 0.05, 0, 1)


def pm_baseline(urban, month):
    """Baseline PM2.5 (ug/m3): urban-driven, worse in winter, cleaner in monsoon."""
    seasonal = {6: 0.7, 7: 0.55, 8: 0.55, 9: 0.7,
                11: 1.3, 12: 1.4, 1: 1.4, 2: 1.2}.get(int(month), 1.0)
    return np.clip((25 + 140 * np.asarray(urban)) * seasonal, 5, 450)
