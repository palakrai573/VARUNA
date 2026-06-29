"""
Inference for ClimateUNet: anomalies -> real-world fields.

    pred_anom_scaled = model(input_window)
    field[v, lead] = clim[v, doy(forecast_day)] + pred_anom_scaled[v,lead]*std[v]
    rain is clipped to >= 0.

Returns forecasts in real units (mm/day, deg C) on the national grid.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models.architecture import build_model  # noqa: E402
from models import dataset as D  # noqa: E402


class Forecaster:
    def __init__(self, ckpt=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model().to(self.device).eval()
        ckpt = ckpt or os.path.join(C.CKPT_DIR, "climate_unet.pt")
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["state_dict"])

    @torch.no_grad()
    def predict(self, cube, t, carr, std, dates):
        """Forecast HORIZON days from forecast-start index t.

        Returns dict: frames {var: (HORIZON,H,W) real units}, dates list.
        """
        X, _ = D._window(cube, t)                       # (INPUT_DAYS*3,H,W)
        xb = torch.from_numpy(X[None]).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            out = self.model(xb)[0].float().cpu().numpy()   # (HORIZON*3,H,W)
        H, W = out.shape[1], out.shape[2]
        out = out.reshape(C.HORIZON, 3, H, W)               # (lead,var,H,W)

        f_dates = [dates[t] + pd.Timedelta(days=k) for k in range(C.HORIZON)]
        frames = {v: np.empty((C.HORIZON, H, W), dtype="float32") for v in C.VARIABLES}
        for lead in range(C.HORIZON):
            doy = min(int(f_dates[lead].dayofyear), 365)
            for vi, v in enumerate(C.VARIABLES):
                field = carr[v][doy - 1] + out[lead, vi] * std[vi]
                if v == "rain":
                    field = np.clip(field, 0.0, None)
                frames[v][lead] = field
        return {"frames": frames, "dates": f_dates}


def load_everything(ckpt=None):
    """Convenience: cache + forecaster, ready to predict."""
    obs, clim, stats, landmask, grid = D.load_cache()
    cube, dates, carr, std = D.build_anomaly_cube(obs, clim, stats)
    fc = Forecaster(ckpt)
    return dict(obs=obs, clim=clim, stats=stats, landmask=landmask, grid=grid,
                cube=cube, dates=dates, carr=carr, std=std, forecaster=fc)
