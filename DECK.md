# AI-Powered Digital Twin of India's Climate — VARUNA
### Virtual AI Replica for Understanding & Nowcasting the Atmosphere
### ISRO Bharatiya Antariksh Hackathon (BAH) 2026 · Problem Statement #5
*An indigenous AI climate twin built entirely on real national (IMD) data — trained, evaluated and served in one reproducible repo.*

> This deck is organised to map **1:1 onto the eight evaluation parameters**. Numbers are auto-filled from `outputs/eval_metrics.json` and refreshed by `tools/build_deck.py`.

---

## Slide 1 — The problem & our framing  *(Problem Understanding & Clarity)*
- PS#5 asks for a **dynamic virtual replica of India's climate** that ingests observations, forecasts rainfall & temperature, drives **connected applications**, and supports **what-if** analysis.
- Our scope: a **national** twin (not just a pilot), forecasting rain + max/min temperature, with **hazard early-warning**, **assimilation**, and **urban-heat / air-quality** applications.
- Honest by design: every claim is backed by a script; nothing is synthetic.

## Slide 2 — Real national data  *(Data Usage & Pre-processing)*
- **IMD gridded**: rainfall 0.25° (135×129), max/min temperature 1.0° (31×31), **2010–2024**, via `imdlib`.
- Pipeline: temperature **regridded** to the 0.25° grid → land mask → **smoothed day-of-year climatology (train years only)** → **anomaly** cube → no-leakage split (train 2010–21 · val 22 · test 23–24).
- **INSAT/MOSDAC** satellite ingestion built in (real HDF5 loader; bring-your-own product).

## Slide 3 — The AI model  *(Model Development & Technical Approach)*
- **ClimateUNet**: residual U-Net + squeeze-excite attention, **predicts anomalies** vs climatology and **refines a persistence-of-anomaly prior** (residual skip → can only match-or-beat the baseline).
- **Direct multi-horizon** (10 lead days in one pass) → **no autoregressive divergence** (the old ConvLSTM hit ±6σ & negative rainfall by day 21; ours stays physical).
- **XGBoost** complementary station model; **CNN+XGB ensemble** at city scale. GPU-trained (RTX 4050).

## Slide 4 — It works, and we prove it  *(Prediction Performance & Validation)*
- Weather-standard metrics on **unseen 2023–2024**, land-masked + area-weighted, per lead day:
  **RMSE / MAE (mm, °C), Anomaly Correlation (ACC), skill vs persistence & persistence-of-anomaly.**
- Headline (day-1): **rain ≈ {RAIN_RMSE} mm**, **tmax ≈ {TMAX_RMSE} °C (ACC {TMAX_ACC})**, **tmin ≈ {TMIN_RMSE} °C (ACC {TMIN_ACC})**.
- **Beats persistence by {SKILL_PERS}%**; **beats persistence-of-anomaly** after residual upgrade.

## Slide 5 — A true digital twin  *(Digital Twin Concept Implementation)*
- Continuously-evolving **state** = AI forecast **fused with observations** via **Optimal Interpolation** (correlated background-error covariance — beyond nudging).
- Forecast → state → hazards → scenarios form a closed, interactive loop.

## Slide 6 — Visualization & UI  *(Visualization & User Interface)*
- Dark "orbital" dashboard: interactive **national + pilot** raster maps on a dark basemap.
- Views: Climate Twin · **Hazards & Extremes** · What-if Simulator · Validation & Skill · Satellite.

## Slide 7 — Innovation  *(Innovation & Creativity)*
- **Hazard early-warning from the forecast**: heatwave (IMD departure criteria), heavy-rain categories, dry-spell/drought.
- **Residual-over-baseline** learning, **physics-informed** heat/AQI proxies, **two-model ensemble**, real **satellite-ready** ingestion.

## Slide 8 — Scale-up & close  *(Presentation & Communication)*
- Scalable to national operations: foundation models (**Prithvi-WxC / Pangu-Weather on IMDAA/BharatBench**), live **INSAT** feeds, real EnKF/4D-Var assimilation.
- Everything reproducible: `python data/prepare.py → models/train.py → evaluation/evaluate.py → streamlit run app.py`.

---
*Atmanirbhar climate intelligence on India's own data.*
