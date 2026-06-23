# TraffiX

A predictive traffic management dashboard built for the Bengaluru Traffic Police (BTP), designed as an operational extension to the ASTRAM Intelligence SDK. TraffiX moves beyond reactive incident mapping by using machine learning to forecast event clearance windows and prescribing mathematically-derived resource allocation plans before congestion develops.

---

## Features

**Dual-Path ML Clearance and Severity Forecasting**
The training set is expanded from 2,460 to 3,515 usable rows via timestamp imputation from downstream `resolved_datetime` and `modified_datetime` fields. Planned events (processions, protests, VIP movements) are handled by a specialist model path to prevent feature space contamination by routine vehicle breakdowns. A calibrated Random Forest classifier predicts structural severity using only features available before operator assignment, eliminating data leakage. A bilingual NLP engine (`TrafficNLPEngine`) extracts Kannada and English signals from commentary fields, covering approximately 83% of described incidents.

**MILP-Optimal Resource Allocation**
`scipy.optimize.milp` evaluates each incident profile and prescribes an exact field kit: foot patrols, strike teams, and physical barricade counts. The optimizer minimizes municipal operational cost while ensuring combined personnel absorption rates meet or exceed the predicted delay impact.

**Spatial Network Routing via OSMnx**
Real road approaches within a 350m radius of any GPS coordinate in Bengaluru are resolved through an OpenStreetMap ego-graph traversal. Blocked nodes are dropped and an edge-penalization shortest-path algorithm computes genuine alternate travel vectors. A live matrix integration with OpenRouteService monitors congestion across 12 primary BTP corridors when an API key is provided.

**Field Utilities**
An on-demand PDF dispatch card is generated in BTP Sub-Inspector format, containing plain-language constable instructions, dynamic emergency lanes, and resource counts. Stalled events (open more than 2x their predicted clearance time) are flagged automatically. A JSON-based post-event feedback loop accumulates MAE margins across shifts for real-world accuracy tracking.

---

## Architecture

```
TraffiX-main/
├── data/
│   ├── Astram_event_data_anonymized_*.csv   # ASTRAM incident dataset
│   ├── engineered_traffic_features.csv      # Pre-built feature store
│   ├── bengaluru_drive_graph.pkl            # Cached OSM road network
│   └── traffix_model.pkl                    # Trained ML model bundle
└── src/
    ├── app.py                # Streamlit dashboard (entry point)
    ├── prediction_engine.py  # ML training, inference, stall detection
    ├── graph_engine.py       # OSMnx spatial snapping and diversion routing
    ├── optimizer.py          # MILP resource allocation model
    ├── live_traffic.py       # ORS API corridor monitoring
    ├── barricade_engine.py   # Physical asset placement logic
    ├── dispatch_card.py      # PDF generation (fpdf2)
    ├── nlp_engine.py         # Kannada/English regex NLP
    └── pipeline.py           # Batch feature engineering pipeline
```

---

## Requirements

- Python 3.10 or later
- An OpenRouteService API key for live corridor data (optional — the dashboard runs in simulation mode without one)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/TraffiX.git
cd TraffiX
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

The core dependencies are: `pandas`, `numpy`, `networkx`, `osmnx`, `pyproj`, `scikit-learn`, `imbalanced-learn`, `scipy`, `streamlit`, `streamlit-folium`, `folium`, and `fpdf2`.

**4. Verify data files are in place**

The following files must exist under `data/` before launching:

```
data/Astram_event_data_anonymized_-_Astram_event_data_anonymizedb40ac87.csv
data/engineered_traffic_features.csv
data/bengaluru_drive_graph.pkl
data/traffix_model.pkl
```

These are included in the repository. If the ASTRAM CSV is missing, the dashboard will be unable to load incident data. If `bengaluru_drive_graph.pkl` is missing, the app will fall back to cardinal/hardcoded barricade placement instead of real OSM road traversal.

---

## Running the Dashboard

```bash
cd src
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

**ORS API key (optional)**

To enable live corridor traffic data, either set the environment variable before launching:

```bash
export ORS_API_KEY=your_key_here   # Linux / macOS
set ORS_API_KEY=your_key_here      # Windows
streamlit run app.py
```

Or paste the key into the "API Configuration" expander in the sidebar after launch. Without a key, the dashboard runs in simulation mode with pre-computed corridor data. A free key can be obtained at [openrouteservice.org](https://openrouteservice.org).

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| Live Traffic Map | Corridor speed/congestion heatmap, ORS live or simulation data, diversion suggestions |
| Barricade Placement | OSM-derived approach roads, physical asset placement, printable PDF dispatch card |
| Event Prediction | ML clearance forecast, MILP resource allocation, stalled event flags |
| DNA Matching | (Feature tab) |
| Compliance Risk | (Feature tab) |
| Analytics | Historical learning set, junction repeat-offender table, post-event feedback logging, MAE tracking |

**OSM Diversion Routing toggle**

The sidebar contains an "OSM Diversion Routing" checkbox. Enabling it downloads and caches the Bengaluru road network on first run (approximately 1-2 minutes). Subsequent runs load instantly from the cached `.pkl` file. Enable this at least once before a demo to warm the cache.

---

## Rebuilding the Feature Store

If you update the ASTRAM CSV and need to regenerate `engineered_traffic_features.csv`:

```bash
cd src
python pipeline.py
```

This runs the full NLP extraction, spatial snapping, and feature engineering pipeline and writes the output back to `data/`.

---

## Dataset

The ASTRAM dataset (`Astram_event_data_anonymized`) contains anonymized incident records from Bengaluru traffic operations. Key fields used: `start_datetime`, `closed_datetime`, `resolved_datetime`, `modified_datetime`, `event_type`, `event_cause`, `priority`, `corridor`, `zone`, `latitude`, `longitude`, `comment`, `description`.

---
