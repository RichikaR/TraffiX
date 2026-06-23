# 🚦 TraffiX = Production Dashboard for Bengaluru Traffic Police

TraffiX is a high-fidelity, predictive traffic management dashboard designed as an extension to the **ASTRAM Intelligence SDK**. Engineered specifically for the Bengaluru Traffic Police (BTP), it bridges the gap between raw incident logs and actionable field deployments[cite: 11]. Moving beyond purely reactive mapping, TraffiX uses Machine Learning to forecast event clearance windows and prescribes mathematical resource-allocation plans before congestion spikes[cite: 11, 19].

Built with a clean, professional Swiss/Bauhaus-inspired light theme optimized for quick data scanability[cite: 11].

---

##  Core Features

### 1. Dual-Path ML Clearance & Severity Forecasting
* **Target Imputation Strategy:** Enhanced the training set slice by 43% (expanding usable rows from 2460 to 3515) by robustly resolving missing timestamps via downstream `resolved_datetime` and `modified_datetime` metrics[cite: 19].
* **Planned vs. Unplanned Event Split:** Utilizes a specialist crowd-event model for high-impact public gatherings (processions, protests, VIP movements) to completely eliminate feature space domination by routine vehicle breakdowns[cite: 19].
* **Leakage-Free Calibrated Classifier:** Predicts structural operational severity using a calibrated Random Forest pipeline that excludes target indicators knowable only *after* operator assignment, solving critical data leakage bugs[cite: 19].
* **Text Mining (NLP Coverage):** Integrates a regex-driven Kannada/English text extraction tool (`TrafficNLPEngine`) capturing 83% of commentary details to flag road block types and designate city clearance agencies[cite: 16, 19].

### 2.  MILP-Optimal Resource Allocation
* **Cost-Minimized Provisioning:** Powered by `scipy.optimize.milp` to evaluate incident profiles and prescribe an exact field kit (Foot patrols, Strike teams, physical barricades)[cite: 17]. 
* Minimizes municipal operational costs while mathematically ensuring combined personnel mitigation absorbing rates meet or exceed predicted delay impacts[cite: 17].

### 3. Spatial Network Graph Traversals (OSMnx & OpenStreetMap)
* **Real-Road Approach Detection:** Traverses an OpenStreetMap ego-graph drive network natively inside a 350m incident radius to pull true road approaches and bearings for any raw GPS coordinate across Bengaluru[cite: 12, 14].
* **Zone-Isolated Diversion Routing:** Dynamically drops blocked nodes from local routable graphs and applies an edge-penalization shortest-path mechanism (`networkx`) to render real alternate travel vectors around gridlocks[cite: 14].
* Includes an out-of-the-box live matrix integration with **OpenRouteService** to monitor real travel congestion across 12 primary BTP corridors[cite: 11, 15].

### 4. Operations-Ready Field Utilities
* **Sub-Inspector Field Dispatch Card:** Generates an on-demand, print-ready field order template containing plain-language constable instructions, dynamic emergency lanes, and auto-generated PDFs via `fpdf2`[cite: 13].
* **Stalled Event Architecture:** Scans live records to flag active anomalies remaining open over 2× their predicted ML clearance window[cite: 11, 19].
* **Historical Learning Loops:** Houses a JSON-based post-event telemetry feedback channel that accumulates Mean Absolute Error (MAE) margins over consecutive active shifts to track real-world accuracy[cite: 11, 19].

---

##  System Architecture & Modules

* **`app.py`:** Core dashboard engine organizing metrics, UI elements, live folium maps, and analytic heatmap structures[cite: 11].
* **`prediction_engine.py`:** Training frameworks, dataset imputation, data quality matrix handlers, and inference engines[cite: 19].
* **`graph_engine.py`:** Projected coordinate transformations, network snapping logic, and dual-alternative route optimization[cite: 14].
* **`optimizer.py`:** Scipy Mixed-Integer Linear Programming models containing operational duty costs[cite: 17].
* **`live_traffic.py`:** Live API clients mapping corridor travel velocities and congestion ratios against night baseline constants[cite: 15].
* **`barricade_engine.py`:** Structural placement logic deploying physical assets onto OSM approaches or pre-verified junction lookups[cite: 12].
* **`dispatch_card.py`:** Print layout rendering code generating Latin-1 secure PDF bitstreams[cite: 13].
* **`nlp_engine.py`:** Multi-lingual regex token pattern processors parsing case severity signals[cite: 16].
* **`pipeline.py`:** Batch data engineering pipeline forming definitive feature store matrix outputs[cite: 18].

---
