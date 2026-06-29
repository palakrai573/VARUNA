# Demo walkthrough (2–3 minutes)

A click-by-click script for the live demo / screen recording. Maps each step to the evaluation parameter it showcases.

**Setup:** `streamlit run app.py` → open the dashboard. Keep the sidebar visible.

---

### 0:00 — Hook (Problem Understanding)
> "This is a digital twin of India's climate, trained entirely on real IMD data — no synthetic data. One AI model drives forecasts, hazard warnings, and what-if planning across the whole country."
- Point to the header badge: **IMD gridded · 2010–2024 · real only**, and the **day-1 skill vs persistence** KPI.

### 0:20 — Climate Twin (Data + Digital Twin Concept)
- View **🌍 Climate Twin**. Move the **date** slider into 2024 ("unseen test year ✓").
- Switch the **climate layer** (Rainfall / Max temp). Move the **lead day** slider — the national AI forecast updates.
- Right panel: **Optimal Interpolation** — slide the correlation length; show **RMSE-to-obs drops after assimilation** (model + observations fused).

### 0:55 — Hazards & Extremes (Innovation)
- View **🌡️ Hazards & Extremes** → **🔥 Heatwave risk**. Show "% area under heatwave" forecast days ahead.
- Switch to **🌧️ Heavy-rain category** and **🏜️ Dry-spell length** — early-warning for monsoon and drought.

### 1:30 — What-if Simulator (Connected Applications)
- View **🧪 What-if Simulator**. Pick **South India (pilot zoom)**.
- Raise **🌳 greening** and **🏠 cool-roof albedo**; toggle **Heat-stress** / **AQI** / **Cooling** layers.
- Read the metrics: **peak surface cooling**, **AQI band**, **heat-danger area reduced vs baseline**.

### 2:10 — Validation & Skill (Prediction Performance)
- View **📈 Validation & Skill**: per-variable **RMSE / ACC / skill** on unseen years; the **skill-vs-lead** curves.
- Pick a city → show **observed → ClimateUNet → CNN+XGB ensemble** forecast lines.

### 2:40 — Close (Presentation / Scale-up)
- View **🛰️ Satellite (INSAT)** — real MOSDAC ingestion ready.
- > "Real data, a model that beats the operational baselines, connected climate apps, and a clear path to foundation models and live INSAT feeds. Atmanirbhar climate intelligence."

---
**Recording tip:** 1280×720, hide the Streamlit menu (`Settings → wide mode`), and pre-run one forecast so maps are cached for smooth playback.
