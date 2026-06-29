# AI-Powered Digital Twin of India's Climate â€” VARUNA
### Virtual AI Replica for Understanding & Nowcasting the Atmosphere
### ISRO Bharatiya Antariksh Hackathon (BAH) 2026 Â· Problem Statement #5
*An indigenous AI climate twin built entirely on real national (IMD) data â€” trained, evaluated and served in one reproducible repo.*

> This deck is organised to map **1:1 onto the eight evaluation parameters**. Numbers are auto-filled from `outputs/eval_metrics.json` and refreshed by `tools/build_deck.py`.

---

## Slide 1 â€” The problem & our framing  *(Problem Understanding & Clarity)*
- PS#5 asks for a **dynamic virtual replica of India's climate** that ingests observations, forecasts rainfall & temperature, drives **connected applications**, and supports **what-if** analysis.
- Our scope: a **national** twin (not just a pilot), forecasting rain + max/min temperature, with **hazard early-warning**, **assimilation**, and **urban-heat / air-quality** applications.
- Honest by design: every claim is backed by a script; nothing is synthetic.

## Slide 2 â€” Real national data  *(Data Usage & Pre-processing)*
- **IMD gridded**: rainfall 0.25Â° (135Ã—129), max/min temperature 1.0Â° (31Ã—31), **2010â€“2024**, via `imdlib`.
- Pipeline: temperature **regridded** to the 0.25Â° grid â†’ land mask â†’ **smoothed day-of-year climatology (train years only)** â†’ **anomaly** cube â†’ no-leakage split (train 2010â€“21 Â· val 22 Â· test 23â€“24).
- **INSAT/MOSDAC** satellite ingestion built in (real HDF5 loader; bring-your-own product).

## Slide 3 â€” The AI model  *(Model Development & Technical Approach)*
- **ClimateUNet**: residual U-Net + squeeze-excite attention, **predicts anomalies** vs climatology and **refines a persistence-of-anomaly prior** (residual skip â†’ can only match-or-beat the baseline).
- **Direct multi-horizon** (10 lead days in one pass) â†’ **no autoregressive divergence** (the old ConvLSTM hit Â±6Ïƒ & negative rainfall by day 21; ours stays physical).
- **XGBoost** complementary station model; **CNN+XGB ensemble** at city scale. GPU-trained (RTX 4050).

## Slide 4 â€” It works, and we prove it  *(Prediction Performance & Validation)*
- Weather-standard metrics on **unseen 2023â€“2024**, land-masked + area-weighted, per lead day:
  **RMSE / MAE (mm, Â°C), Anomaly Correlation (ACC), skill vs persistence & persistence-of-anomaly.**
- Headline (day-1): **rain â‰ˆ 9.9 mm**, **tmax â‰ˆ 1.20 Â°C (ACC 0.76)**, **tmin â‰ˆ 0.84 Â°C (ACC 0.76)**.
- **Beats persistence by 12%**; **beats persistence-of-anomaly** after residual upgrade.

## Slide 5 â€” A true digital twin  *(Digital Twin Concept Implementation)*
- Continuously-evolving **state** = AI forecast **fused with observations** via **Optimal Interpolation** (correlated background-error covariance â€” beyond nudging).
- Forecast â†’ state â†’ hazards â†’ scenarios form a closed, interactive loop.

## Slide 6 â€” Visualization & UI  *(Visualization & User Interface)*
- Dark "orbital" dashboard: interactive **national + pilot** raster maps on a dark basemap.
- Views: Climate Twin Â· **Hazards & Extremes** Â· What-if Simulator Â· Validation & Skill Â· Satellite.

## Slide 7 â€” Innovation  *(Innovation & Creativity)*
- **Hazard early-warning from the forecast**: heatwave (IMD departure criteria), heavy-rain categories, dry-spell/drought.
- **Residual-over-baseline** learning, **physics-informed** heat/AQI proxies, **two-model ensemble**, real **satellite-ready** ingestion.

## Slide 8 â€” Scale-up & close  *(Presentation & Communication)*
- Scalable to national operations: foundation models (**Prithvi-WxC / Pangu-Weather on IMDAA/BharatBench**), live **INSAT** feeds, real EnKF/4D-Var assimilation.
- Everything reproducible: `python data/prepare.py â†’ models/train.py â†’ evaluation/evaluate.py â†’ streamlit run app.py`.

---
*Atmanirbhar climate intelligence on India's own data.*
