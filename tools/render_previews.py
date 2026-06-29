"""
Render static preview PNGs of AI national forecasts for the README / deck.
Run after training:  python tools/render_previews.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models.forecast import load_everything  # noqa: E402
from viz import theme  # noqa: E402

PREVIEW_DIR = os.path.join(C.OUTPUTS_DIR, "preview")
os.makedirs(PREVIEW_DIR, exist_ok=True)


def _panel(ax, field, lats, lons, key, landmask, title):
    f = np.where(landmask, field, np.nan)
    vmin, vmax = theme.VRANGE[key]
    im = ax.pcolormesh(lons, lats, f, cmap=theme.CMAPS[key], vmin=vmin, vmax=vmax,
                       shading="auto")
    ax.set_title(title, color="#e9eeff", fontsize=11)
    ax.set_facecolor("#04070f")
    ax.tick_params(colors="#8b97c6", labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)


def main():
    B = load_everything()
    lats, lons = B["grid"]["lat"], B["grid"]["lon"]
    lm = B["landmask"]
    t = next(i for i in range(len(B["dates"]) - C.HORIZON, 0, -1)
             if B["dates"][i].year in C.TEST_YEARS)
    fr = B["forecaster"].predict(B["cube"], t, B["carr"], B["std"], B["dates"])
    frames, fdates = fr["frames"], fr["dates"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#04070f")
    _panel(axes[0], frames["rain"][0], lats, lons, "rain", lm,
           f"AI Rainfall · {fdates[0].strftime('%d %b %Y')}")
    _panel(axes[1], frames["tmax"][0], lats, lons, "tmax", lm, "AI Max Temp")
    _panel(axes[2], frames["tmin"][0], lats, lons, "tmin", lm, "AI Min Temp")
    fig.tight_layout()
    fig.savefig(os.path.join(PREVIEW_DIR, "national_forecast.png"), dpi=130,
                facecolor="#04070f")
    print("saved", os.path.join(PREVIEW_DIR, "national_forecast.png"))


if __name__ == "__main__":
    main()
