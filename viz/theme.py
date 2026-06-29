"""
Visual identity for the AI Digital Twin dashboard.

A dark "orbital" theme with custom thermal / rainfall colormaps designed to read
well over a dark basemap. Pure-Python (matplotlib), no API tokens.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

# Free Carto dark basemap tiles (no token required).
DARK_TILES = "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
DARK_ATTR = "&copy; OpenStreetMap &copy; CARTO"

RAIN = LinearSegmentedColormap.from_list(
    "rain", ["#0a1230", "#0E9AA7", "#22D3EE", "#3B82F6", "#7C3AED", "#F0ABFC"])
TEMP = LinearSegmentedColormap.from_list(
    "temp", ["#1E5BFF", "#22D3EE", "#34D399", "#FDE047", "#FB923C", "#EF4444"])
COOLING = LinearSegmentedColormap.from_list(
    "cooling", ["#0a1230", "#0E9AA7", "#22D3EE", "#5EEAD4", "#A7F3D0"])
AQI = LinearSegmentedColormap.from_list(
    "aqi", ["#2DC937", "#A3C614", "#FFD60A", "#FF7B00", "#FF2D55", "#7E0023"])

CMAPS = {"rain": RAIN, "tmax": TEMP, "tmin": TEMP,
         "heat_index": TEMP, "lst": TEMP, "cooling": COOLING, "aqi": AQI}

VRANGE = {"rain": (0, 60), "tmax": (15, 48), "tmin": (5, 35),
          "heat_index": (20, 55), "lst": (20, 55), "cooling": (-2, 8), "aqi": (0, 400)}

LABELS = {"rain": "Rainfall (mm/day)", "tmax": "Max temp (°C)", "tmin": "Min temp (°C)",
          "heat_index": "Heat index (°C)", "aqi": "Air Quality Index", "cooling": "Cooling (°C)"}


def streamlit_css() -> str:
    p = C.PALETTE
    return f"""
    <style>
      .stApp {{ background: radial-gradient(1200px 600px at 72% -12%, {p['bg2']} 0%, {p['bg']} 60%);
                color: {p['text']}; }}
      section[data-testid="stSidebar"] {{ background: {p['panel']}; }}
      h1,h2,h3 {{ color: {p['text']}; letter-spacing:.3px; }}
      .tn-tag {{ color: {p['accent']}; font-weight:600; }}
      .tn-card {{ background:{p['panel']}; border:1px solid #1d2a5a; border-radius:14px;
                  padding:12px 16px; margin-bottom:8px; }}
      .tn-card .v {{ font-size:1.5rem; font-weight:700; color:{p['accent']}; }}
      .tn-card .l {{ color:{p['muted']}; font-size:.78rem; text-transform:uppercase; letter-spacing:.5px; }}
      div[data-testid="stMetricValue"] {{ color:{p['accent']}; }}
    </style>
    """
