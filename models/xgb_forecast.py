"""
Gradient-boosted (XGBoost) station-level forecaster - a second, complementary
modelling approach alongside the ClimateUNet spatial model.

Where ClimateUNet predicts full spatial fields, XGBoost excels at point/station
forecasting from engineered tabular features (lagged anomalies, climatology,
seasonality, location, recent trend, POA prior). We train one GPU-accelerated
model per variable that predicts the scaled anomaly at any lead day, then
ensemble it with ClimateUNet at city scale.

    python models/xgb_forecast.py            # train + evaluate + save
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models import dataset as D  # noqa: E402

N_CELLS = 350          # random land cells sampled for the tabular set
TRAIN_STRIDE = 5       # subsample training windows
RNG = np.random.default_rng(0)


def _features(cube, dates, carr, std, lat, lon, t, iy, ix, lead):
    """Feature row for one (window t, cell, lead)."""
    doy = min(int((dates[t] + np.timedelta64(lead + 1, "D")).dayofyear)
              if hasattr(dates[t], "dayofyear") else 1, 366)
    hist = cube[t - C.INPUT_DAYS:t, iy, ix, :]          # (7,3)
    last = cube[t - 1, iy, ix, :]                        # (3,)
    mean7 = hist.mean(0)                                 # (3,)
    slope = hist[-1] - hist[0]                           # (3,)
    rho = np.array([C.RHO[v] for v in C.VARIABLES], dtype="float32")
    poa = rho ** (lead + 1) * last                       # (3,)
    d = (dates[t] + np.timedelta64(lead, "D"))
    doy = min(int(getattr(d, "dayofyear", 1)), 366)
    feat = np.concatenate([
        hist.reshape(-1), mean7, slope, poa, last,
        [np.sin(2 * np.pi * doy / 366), np.cos(2 * np.pi * doy / 366)],
        [(lat[iy] - 6.5) / 32.0, (lon[ix] - 66.5) / 34.0, lead / C.HORIZON],
    ]).astype("float32")
    return feat


def build_tabular(cube, dates, carr, std, landmask, grid, idx, n_cells=N_CELLS, stride=1):
    lat, lon = grid["lat"], grid["lon"]
    land_cells = np.argwhere(landmask)
    sel = land_cells[RNG.choice(len(land_cells), size=min(n_cells, len(land_cells)), replace=False)]
    idx = idx[::stride]
    X, Y = [], {v: [] for v in C.VARIABLES}
    import pandas as pd
    dts = pd.to_datetime(dates)
    for t in idx:
        t = int(t)
        for (iy, ix) in sel:
            hist = cube[t - C.INPUT_DAYS:t, iy, ix, :]
            last = cube[t - 1, iy, ix, :]
            mean7 = hist.mean(0); slope = hist[-1] - hist[0]
            rho = np.array([C.RHO[v] for v in C.VARIABLES], dtype="float32")
            for lead in range(C.HORIZON):
                poa = rho ** (lead + 1) * last
                doy = min(int(dts[t].dayofyear) + lead, 366)
                feat = np.concatenate([
                    hist.reshape(-1), mean7, slope, poa, last,
                    [np.sin(2*np.pi*doy/366), np.cos(2*np.pi*doy/366)],
                    [(lat[iy]-6.5)/32.0, (lon[ix]-66.5)/34.0, lead/C.HORIZON],
                ]).astype("float32")
                X.append(feat)
                for vi, v in enumerate(C.VARIABLES):
                    Y[v].append(cube[t + lead, iy, ix, vi])
    X = np.asarray(X, dtype="float32")
    Y = {v: np.asarray(Y[v], dtype="float32") for v in C.VARIABLES}
    return X, Y


def train():
    import xgboost as xgb
    obs, clim, stats, landmask, grid = D.load_cache()
    cube, dates, carr, std = D.build_anomaly_cube(obs, clim, stats)
    sp = D.split_indices(dates)
    print("[xgb] building tabular features…", flush=True)
    Xtr, Ytr = build_tabular(cube, dates, carr, std, landmask, grid, sp["train"], stride=TRAIN_STRIDE)
    Xva, Yva = build_tabular(cube, dates, carr, std, landmask, grid, sp["val"], n_cells=200, stride=3)
    print(f"[xgb] train rows {Xtr.shape}, val rows {Xva.shape}", flush=True)

    params = dict(n_estimators=600, max_depth=8, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=2.0, tree_method="hist", device="cuda")
    models = {}
    report = {}
    for vi, v in enumerate(C.VARIABLES):
        print(f"[xgb] training {v}…", flush=True)
        m = xgb.XGBRegressor(**params, early_stopping_rounds=30, eval_metric="rmse")
        m.fit(Xtr, Ytr[v], eval_set=[(Xva, Yva[v])], verbose=False)
        models[v] = m
        # val RMSE in real units (scaled anomaly * std)
        pred = m.predict(Xva)
        rmse = float(np.sqrt(np.mean(((pred - Yva[v]) * std[vi]) ** 2)))
        report[v] = {"val_rmse_real": rmse, "best_iter": int(m.best_iteration)}
        m.save_model(os.path.join(C.CKPT_DIR, f"xgb_{v}.json"))
        print(f"   {v}: val RMSE {rmse:.3f} (real units), best_iter {m.best_iteration}", flush=True)

    with open(os.path.join(C.OUTPUTS_DIR, "xgb_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("[xgb] saved models to models/checkpoints/xgb_*.json")
    return report


class CityForecaster:
    """Load XGBoost models and forecast a single grid cell over the horizon.

    Uses the native Booster API (not the sklearn wrapper) to stay compatible
    across xgboost / scikit-learn versions.
    """

    def __init__(self):
        import xgboost as xgb
        self.models = {}
        for v in C.VARIABLES:
            bst = xgb.Booster()
            bst.load_model(os.path.join(C.CKPT_DIR, f"xgb_{v}.json"))
            self.models[v] = bst

    def predict_cell(self, cube, dates, carr, std, grid, t, iy, ix):
        import pandas as pd
        import xgboost as xgb
        lat, lon = grid["lat"], grid["lon"]
        dts = pd.to_datetime(dates)
        hist = cube[t - C.INPUT_DAYS:t, iy, ix, :]
        last = cube[t - 1, iy, ix, :]
        mean7 = hist.mean(0); slope = hist[-1] - hist[0]
        rho = np.array([C.RHO[v] for v in C.VARIABLES], dtype="float32")
        rows = []
        for lead in range(C.HORIZON):
            poa = rho ** (lead + 1) * last
            doy = min(int(dts[t].dayofyear) + lead, 366)
            rows.append(np.concatenate([
                hist.reshape(-1), mean7, slope, poa, last,
                [np.sin(2*np.pi*doy/366), np.cos(2*np.pi*doy/366)],
                [(lat[iy]-6.5)/32.0, (lon[ix]-66.5)/34.0, lead/C.HORIZON]]).astype("float32"))
        dmat = xgb.DMatrix(np.asarray(rows, dtype="float32"))
        out = {}
        for vi, v in enumerate(C.VARIABLES):
            anom = self.models[v].predict(dmat)
            doys = [min(int(dts[t].dayofyear) + k, 365) for k in range(C.HORIZON)]
            field = np.array([carr[v][doys[k] - 1, iy, ix] + anom[k] * std[vi]
                              for k in range(C.HORIZON)])
            out[v] = np.clip(field, 0, None) if v == "rain" else field
        return out


if __name__ == "__main__":
    train()
