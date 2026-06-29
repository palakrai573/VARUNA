"""
Central configuration for the AI Digital Twin of India's Climate.
ISRO Bharatiya Antariksh Hackathon (BAH) 2026 - Problem Statement #5.

Every module (data, models, evaluation, scenario, twin, viz, app) imports grids,
regions, palette and paths from here so the whole system stays consistent.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")            # imdlib .grd cache
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # NetCDF cache + tensors
MODELS_DIR = os.path.join(ROOT, "models")
CKPT_DIR = os.path.join(MODELS_DIR, "checkpoints")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
ASSETS_DIR = os.path.join(ROOT, "assets")
for _d in (RAW_DIR, PROCESSED_DIR, CKPT_DIR, OUTPUTS_DIR, ASSETS_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- data span
# IMD products are published with a lag; 2024 is the last complete year.
# Full modern record (1981+): dense station coverage, current climate regime,
# and ~3x the data of a short window - which is what cures overfitting.
TRAIN_YEARS = list(range(1981, 2019))   # 1981-2018 (38 yrs)
VAL_YEARS = [2019, 2020]
TEST_YEARS = [2021, 2022, 2023, 2024]   # 4 held-out years -> robust evaluation
ALL_YEARS = TRAIN_YEARS + VAL_YEARS + TEST_YEARS
CLIM_YEARS = TRAIN_YEARS                # climatology from training years only (no leakage)

# --------------------------------------------------------------------------- native IMD grids
# Rainfall: 0.25 deg, 135 lon x 129 lat, 66.5-100.0E / 6.5-38.5N  (the working grid).
RAIN_RES = 0.25
RAIN_LON = (66.5, 100.0)
RAIN_LAT = (6.5, 38.5)
GRID_NLAT = 129
GRID_NLON = 135
# Temperature: 1.0 deg, 31 x 31, 67.5-97.5E / 7.5-37.5N -> regridded up to the rain grid.
TEMP_RES = 1.0
TEMP_LON = (67.5, 97.5)
TEMP_LAT = (7.5, 37.5)

VARIABLES = ["rain", "tmax", "tmin"]
IMD_FILL = (-999.0, 99.9, -99.9)        # imdlib missing/fill sentinels

# --------------------------------------------------------------------------- sequence geometry
INPUT_DAYS = 7        # days of history fed to the model
HORIZON = 10          # forecast lead days predicted in one shot (direct multi-horizon)
CLIM_SMOOTH_WINDOW = 21  # day-of-year climatology smoothing window

# Per-variable anomaly persistence used for the persistence-of-anomaly (POA)
# baseline AND as the physics prior the AI model refines (residual learning).
RHO = {"rain": 0.55, "tmax": 0.80, "tmin": 0.80}

# --------------------------------------------------------------------------- regions
INDIA_BOUNDS = dict(lon_min=66.5, lon_max=100.0, lat_min=6.5, lat_max=38.5)
# Pilot deep-dive region (kept from the original project: Tamil Nadu & Kerala).
PILOT_BOUNDS = dict(lon_min=74.0, lon_max=81.0, lat_min=8.0, lat_max=16.0)
REGIONS = {
    "India (national twin)": INDIA_BOUNDS,
    "South India (pilot zoom)": PILOT_BOUNDS,
}

# Reference cities for the forecast / time-series panels.
CITIES = {
    "Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59),
    "Thiruvananthapuram": (8.52, 76.94), "Kochi": (9.93, 76.27),
    "Hyderabad": (17.39, 78.49), "Mumbai": (19.08, 72.88),
    "Delhi": (28.61, 77.21), "Kolkata": (22.57, 88.36),
    "Ahmedabad": (23.03, 72.58),
}

# --------------------------------------------------------------------------- brand palette
PALETTE = {
    "bg": "#04070f", "bg2": "#0a1230", "panel": "#0d1530",
    "text": "#e9eeff", "muted": "#8b97c6", "accent": "#FF7B00",
    "good": "#2DD4BF",
}
# India AQI categories (CPCB).
AQI_BANDS = [
    (0, 50, "Good", "#2DC937"), (51, 100, "Satisfactory", "#A3C614"),
    (101, 200, "Moderate", "#FFD60A"), (201, 300, "Poor", "#FF7B00"),
    (301, 400, "Very Poor", "#FF2D55"), (401, 500, "Severe", "#7E0023"),
]
