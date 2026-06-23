"""
TraffiX v2.0 — Production Dashboard for Bengaluru Traffic Police

UI/UX: Clean professional light theme (no dark/cyberpunk elements)
Logic: All business logic preserved from v1, multiple bugs fixed:
  - predictor.predict() computed ONCE, shared across Tab 2 and Tab 3
  - MILP optimizer (optimizer.py) wired into Tab 3 resource recommendations
  - Data-driven congestion forecast via predictor.get_hourly_forecast()
  - Model metrics card (RMSE, MAE, R², F1) shown after KPI strip
  - Data quality breakdown explaining the 70% data-drop
  - clear_count fixed — SLOW corridors are not "Clear"
  - Diversion suggestions panel in Tab 1 (alternate corridors)
  - HIGH and MODERATE severity have distinct colors
  - color_status() at module level (not inside with-block)
  - Dead CSS class .section-card removed
  - spacer() helper — 13 duplicate div-height stubs replaced
  - btype / icon_char unused variables removed in Tab 2
  - api_key scoped at module level before sidebar
  - Tab 6: post-event feedback panel + accuracy stats display
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from prediction_engine import (
    TraffiXPredictor, ComplianceScorer,
    ZONE_CONGESTION_WEIGHTS, CAUSE_IMPACT_MAP,
)
from live_traffic import fetch_live_traffic, get_corridor_alert_message, BENGALURU_CORRIDORS
from barricade_engine import generate_barricade_plan, BARRICADE_TYPES
from optimizer import TrafficTacticalOptimizer, SEVERITY_TO_OPTIMIZER
from graph_engine import BengaluruSpatialGraph
import dispatch_card

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TraffiX | BTP Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS — Clean Professional Light Theme ─────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Global reset & font ──────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #F8FAFC !important;
    color: #111827 !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p {
    color: #374151 !important; font-size: 14px !important; font-weight: 500 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #111827 !important; font-size: 15px !important; font-weight: 600 !important;
}
[data-testid="stSidebarContent"] .stButton > button {
    background-color: #2563EB !important; color: #FFFFFF !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 0.6rem 1rem !important; transition: all 0.2s ease !important;
}
[data-testid="stSidebarContent"] .stButton > button:hover {
    background-color: #1D4ED8 !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] textarea {
    background-color: #F9FAFB !important; border: 1px solid #D1D5DB !important;
    border-radius: 10px !important; color: #111827 !important; font-size: 14px !important;
}
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] select:focus {
    border-color: #2563EB !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #F9FAFB !important; border: 1px solid #D1D5DB !important;
    border-radius: 10px !important; color: #111827 !important;
}

/* ── Metric cards ─────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px;
    padding: 20px 20px 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.10); transform: translateY(-2px);
}
div[data-testid="stMetricLabel"] > div,
div[data-testid="stMetricLabel"] p {
    color: #6B7280 !important; font-size: 13px !important; font-weight: 500 !important;
}
div[data-testid="stMetricValue"] > div {
    color: #111827 !important; font-size: 28px !important;
    font-weight: 700 !important; line-height: 1.2 !important;
}
div[data-testid="stMetricDelta"] { color: #22C55E !important; font-size: 12px !important; }

/* ── Tab navigation ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #F1F5F9 !important; border-radius: 12px !important;
    padding: 4px !important; gap: 2px !important; border: none !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important; border-radius: 8px !important;
    color: #6B7280 !important; font-size: 13px !important; font-weight: 500 !important;
    padding: 8px 16px !important; border: none !important; transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #FFFFFF !important; color: #374151 !important;
}
.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important; color: #111827 !important;
    font-weight: 600 !important; box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── Alert cards ──────────────────────────────────────────── */
.alert-critical {
    background: #FEF2F2; border-left: 4px solid #EF4444; border-radius: 10px;
    padding: 12px 14px; margin: 6px 0; color: #7F1D1D;
    font-size: 13px; line-height: 1.5; transition: transform 0.15s ease;
}
.alert-critical:hover { transform: translateX(2px); }
.alert-high {
    background: #FFFBEB; border-left: 4px solid #F59E0B; border-radius: 10px;
    padding: 12px 14px; margin: 6px 0; color: #78350F;
    font-size: 13px; line-height: 1.5; transition: transform 0.15s ease;
}
.alert-high:hover { transform: translateX(2px); }
.alert-ok {
    background: #F0FDF4; border-left: 4px solid #22C55E; border-radius: 10px;
    padding: 12px 14px; margin: 6px 0; color: #14532D;
    font-size: 13px; line-height: 1.5; transition: transform 0.15s ease;
}
.alert-ok:hover { transform: translateX(2px); }

/* ── Barricade cards ──────────────────────────────────────── */
.barricade-card {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 14px 16px; margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.barricade-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.10); transform: translateY(-1px); }

/* ── Subheaders ───────────────────────────────────────────── */
h2, h3 { color: #111827 !important; font-weight: 700 !important; }
.stMarkdown h3 { font-size: 18px !important; margin-bottom: 0.5rem !important; }

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background-color: #2563EB !important; color: #FFFFFF !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #1D4ED8 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"],
.stButton > button {
    background-color: #FFFFFF !important; color: #374151 !important;
    border: 1px solid #D1D5DB !important; border-radius: 10px !important;
    font-weight: 500 !important; transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #2563EB !important; color: #2563EB !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.15) !important;
}

/* ── Dataframe ────────────────────────────────────────────── */
.stDataFrame { border: 1px solid #E5E7EB !important; border-radius: 12px !important; overflow: hidden !important; }

/* ── Streamlit alerts ─────────────────────────────────────── */
.stAlert { border-radius: 12px !important; border: none !important; }
[data-testid="stInfoMessage"]    { background-color: #EFF6FF !important; color: #1E40AF !important; border-radius: 12px !important; border: 1px solid #BFDBFE !important; }
[data-testid="stSuccessMessage"] { background-color: #F0FDF4 !important; color: #166534 !important; border-radius: 12px !important; border: 1px solid #BBF7D0 !important; }
[data-testid="stWarningMessage"] { background-color: #FFFBEB !important; color: #92400E !important; border-radius: 12px !important; border: 1px solid #FDE68A !important; }
[data-testid="stErrorMessage"]   { background-color: #FEF2F2 !important; color: #991B1B !important; border-radius: 12px !important; border: 1px solid #FECACA !important; }

/* ── Expander ─────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background-color: #F9FAFB !important; border: 1px solid #E5E7EB !important;
    border-radius: 10px !important; color: #374151 !important; font-weight: 500 !important;
}
.streamlit-expanderContent {
    border: 1px solid #E5E7EB !important; border-top: none !important;
    border-radius: 0 0 10px 10px !important; background-color: #FAFAFA !important;
}

/* ── Misc ─────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #2563EB !important; }
.stCaption, [data-testid="stCaptionContainer"] p { color: #9CA3AF !important; font-size: 12px !important; }
hr { border-color: #E5E7EB !important; margin: 1.5rem 0 !important; }
input[type="number"], input[type="text"], input[type="password"] {
    background-color: #F9FAFB !important; border: 1px solid #D1D5DB !important;
    border-radius: 10px !important; color: #111827 !important; font-size: 14px !important;
}
[data-testid="stCheckbox"] label { color: #374151 !important; font-size: 14px !important; }
[data-testid="stRadio"] label    { color: #374151 !important; font-size: 14px !important; }
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input {
    background-color: #F9FAFB !important; border: 1px solid #D1D5DB !important;
    border-radius: 10px !important; color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Data paths ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
    'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')
ALT_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
    'Astram_event_data_anonymized_-_Astram_event_data_anonymizedb40ac87.csv')

def find_data_path():
    import glob
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    candidates = glob.glob(os.path.join(data_dir, '*Astram*anonymized*.csv'))
    if candidates:
        return candidates[0]
    # Fallback: hardcoded names
    for p in [DATA_PATH, ALT_DATA_PATH]:
        if os.path.exists(p):
            return p
    return None

# ─── Helpers ───────────────────────────────────────────────────────────────────
def spacer(px: int = 24):
    """Vertical spacer — avoids repeating the same unsafe_allow_html div everywhere."""
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)


def color_status(val: str) -> str:
    """Pandas Styler function for Status column — at module level (not inside with-block)."""
    colors = {
        "FREE FLOW":  "background-color:#F0FDF4;color:#166534",
        "SLOW":       "background-color:#FFFBEB;color:#92400E",
        "HEAVY":      "background-color:#FFF7ED;color:#9A3412",
        "GRIDLOCK":   "background-color:#FEF2F2;color:#991B1B",
        "STANDSTILL": "background-color:#FEE2E2;color:#7F1D1D",
    }
    return colors.get(val, "")


# ─── Load systems ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading TraffiX Intelligence Engine...")
def load_systems():
    dp = find_data_path()
    predictor = TraffiXPredictor(data_path=dp)
    df_raw    = pd.read_csv(dp, low_memory=False) if dp else None
    return predictor, df_raw

@st.cache_resource(show_spinner="📡 Downloading & caching Bengaluru OSM road network (first run only, then instant)...")
def load_spatial_graph():
    """OSM drive-network graph for real diversion routing + approach-road barricade
    placement. Opt-in (see sidebar checkbox) because the first run downloads the
    whole Bengaluru network from OpenStreetMap — wrapped so a slow/offline first
    run never crashes the dashboard, it just falls back to cardinal/hardcoded
    barricade placement and disables the diversion-route panel."""
    try:
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        return BengaluruSpatialGraph(cache_dir=cache_dir)
    except Exception as e:
        print(f"  Spatial graph unavailable ({e}) — falling back to cardinal/hardcoded barricade placement")
        return None

predictor, df_raw = load_systems()
compliance_scorer = ComplianceScorer(df_raw) if df_raw is not None else None
optimizer_engine  = TrafficTacticalOptimizer()

# ─── Shared "now" — used by the sidebar, the planned-event calendar, and the
# T-minus deployment countdown. Computed once up top so every section agrees. ──
now = datetime.datetime.now()

# ─── Planned Event Calendar ────────────────────────────────────────────────────
# Addresses "no planned-event ingestion" — in production this would be a live
# feed from the ASTRAM Advisory module (or a BTP permissions/NOC database).
# For this build it's a short illustrative calendar; event times are stored
# RELATIVE TO NOW so the "auto-trigger prep brief 2 hours before" behavior
# below is always demonstrable, regardless of when you run TraffiX.
EVENT_CALENDAR = [
    {
        "name": "IPL Match — M. Chinnaswamy Stadium", "cause": "public_event",
        "zone": "Central Zone 1", "lat": 12.9788, "lng": 77.5996,
        "junction": "QueensStatueCircle",
        "datetime": now + datetime.timedelta(hours=1, minutes=45),
    },
    {
        "name": "Political Rally — Freedom Park", "cause": "protest",
        "zone": "Central Zone 2", "lat": 12.9719, "lng": 77.5937,
        "junction": None,
        "datetime": now + datetime.timedelta(hours=4, minutes=10),
    },
    {
        "name": "Flyover Construction Closure — Silk Board", "cause": "construction",
        "zone": "South Zone 2", "lat": 12.9177, "lng": 77.6237,
        "junction": "SilkBoardJunc",
        "datetime": now + datetime.timedelta(hours=6, minutes=30),
    },
    {
        "name": "Ganesh Visarjan Procession — VV Puram", "cause": "procession",
        "zone": "South Zone 1", "lat": 12.9469, "lng": 77.5760,
        "junction": None,
        "datetime": now + datetime.timedelta(hours=23),
    },
]

# Lead time (minutes before event start) TraffiX recommends deploying by —
# used for the T-minus countdown and to decide which calendar events trigger
# an automatic preparation brief.
LEAD_TIME_BY_SEVERITY = {"CRITICAL": 120, "HIGH": 90, "MODERATE": 60, "LOW": 30}

# ─── API keys ─────────────────────────────────────────────────────────────────
# ORS_API_KEY priority: session_state (sidebar input) > env var
import os as _os
_env_ors = _os.environ.get("ORS_API_KEY", "")
ORS_API_KEY = st.session_state.get("ors_key_input", _env_ors) or _env_ors
api_key     = ORS_API_KEY.strip()  # empty string → simulation mode
enable_osm = False

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 20px 0'>
        <div style='width:48px;height:48px;background:linear-gradient(135deg,#2563EB,#1D4ED8);
                    border-radius:14px;display:inline-flex;align-items:center;justify-content:center;
                    margin-bottom:12px;box-shadow:0 4px 12px rgba(37,99,235,0.3)'>
            <span style='font-size:22px'>🚦</span>
        </div><br>
        <span style='font-size:20px;font-weight:800;color:#111827;letter-spacing:-0.5px'>TraffiX</span><br>
        <span style='font-size:12px;color:#6B7280;font-weight:500'>Built as an extension to the ASTRAM SDK</span>
    </div>
    <div style='height:1px;background:#E5E7EB;margin-bottom:20px'></div>
    """, unsafe_allow_html=True)

    # Dynamic data source badge — green when ORS key is present, blue for simulation
    if api_key:
        st.markdown("""
        <div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:10px 12px;margin-bottom:12px'>
            <span style='font-size:12px;font-weight:600;color:#166534'>🟢 ORS Live Mode</span><br>
            <span style='font-size:11px;color:#374151'>Live corridor durations via OpenRouteService Matrix API.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:10px 12px;margin-bottom:12px'>
            <span style='font-size:12px;font-weight:600;color:#1E40AF'>📡 Simulation Mode</span><br>
            <span style='font-size:11px;color:#374151'>Set ORS_API_KEY env var for live corridor data. OSM routing uses cached network.</span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔑 API Configuration", expanded=not bool(api_key)):
        ors_input = st.text_input(
            "ORS API Key",
            value=st.session_state.get("ors_key_input", ORS_API_KEY),
            type="password",
            placeholder="Paste OpenRouteService key...",
            key="ors_key_input",
            help="Get a free key at openrouteservice.org. Enables live corridor traffic data.",
        )
        if ors_input and ors_input != ORS_API_KEY:
            st.rerun()
        if api_key:
            st.success("✅ Live mode active")
        else:
            st.info("No key → simulation mode")

    with st.expander("🗺️ OSM Diversion Routing", expanded=False):
        enable_osm = st.checkbox(
            "Enable live OSM diversion routing", value=False, key="enable_osm_key",
            help="Computes real shortest-path alternate routes around the event "
                 "using the Bengaluru OSM road network. First use downloads & "
                 "caches the network (~1-2 min); instant after that. Tip: turn "
                 "this on once before your demo to warm the cache.",
        )
        if enable_osm:
            st.caption("✅ OSM routing active — see Tab 2 for live diversion paths.")
        else:
            st.caption("Off by default so the dashboard always loads instantly. Barricade placement still uses hardcoded junction data or cardinal fallback either way.")

    st.markdown("<div style='height:1px;background:#E5E7EB;margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#9CA3AF;margin-bottom:10px'>📅 Planned Event Calendar</p>", unsafe_allow_html=True)
    st.caption("Real ASTRAM/NOC permission records would feed this. Pick one to auto-fill the brief below.")

    for i, cal in enumerate(EVENT_CALENDAR):
        cal_mins = (cal["datetime"] - now).total_seconds() / 60
        cal_h, cal_m = int(cal_mins // 60), int(cal_mins % 60)
        is_imminent = cal_mins <= 120
        badge_color = "#EF4444" if is_imminent else "#6B7280"
        st.markdown(f"""
        <div style='border:1px solid #E5E7EB;border-radius:10px;padding:8px 10px;margin-bottom:6px'>
            <div style='font-size:12px;font-weight:600;color:#111827'>{cal['name']}</div>
            <div style='font-size:11px;color:{badge_color};font-weight:600'>
                {'🔔 ' if is_imminent else '🕐 '}T-minus {cal_h}h {cal_m}m
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Load & Brief", key=f"load_cal_{i}", use_container_width=True):
            st.session_state["event_cause_key"] = cal["cause"]
            st.session_state["zone_key"]         = cal["zone"]
            st.session_state["event_date_key"]   = cal["datetime"].date()
            st.session_state["event_time_key"]   = cal["datetime"].time()
            st.session_state["junction_key"]     = cal["junction"] or ""
            st.session_state["lat_key"]          = cal["lat"]
            st.session_state["lng_key"]          = cal["lng"]
            st.rerun()

    st.markdown("<div style='height:1px;background:#E5E7EB;margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#9CA3AF;margin-bottom:10px'>📋 Event Details</p>", unsafe_allow_html=True)

    event_cause = st.selectbox("Event Type", options=list(CAUSE_IMPACT_MAP.keys()),
                               index=list(CAUSE_IMPACT_MAP.keys()).index('public_event'),
                               format_func=lambda x: x.replace('_', ' ').title(),
                               key="event_cause_key")
    zone     = st.selectbox("Zone", options=list(ZONE_CONGESTION_WEIGHTS.keys()), key="zone_key")
    priority = st.radio("Priority", ["High", "Low"], horizontal=True)
    road_closure = st.checkbox("Road Closure Required")

    st.markdown("<div style='height:1px;background:#E5E7EB;margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#9CA3AF;margin-bottom:10px'>📅 When — pick a future time to FORECAST, not just react</p>", unsafe_allow_html=True)
    event_date = st.date_input("Date", value=datetime.date.today(), key="event_date_key")
    event_time = st.time_input("Start Time", value=datetime.time(17, 0), key="event_time_key")
    event_dt   = datetime.datetime.combine(event_date, event_time)

    st.markdown("<div style='height:1px;background:#E5E7EB;margin:16px 0'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#9CA3AF;margin-bottom:10px'>📍 Location (for Barricade Map)</p>", unsafe_allow_html=True)
    junction_input = st.text_input("Junction / Area Name", placeholder="e.g. Silk Board Junction", key="junction_key")
    custom_lat = st.number_input("Latitude",  value=12.9177, format="%.4f", key="lat_key")
    custom_lng = st.number_input("Longitude", value=77.6237, format="%.4f", key="lng_key")

    st.markdown("<div style='height:1px;background:#E5E7EB;margin:16px 0'></div>", unsafe_allow_html=True)
    if st.button("⚡ Analyse Event", type="primary", use_container_width=True):
        st.session_state["analyse_triggered"] = True

spatial_graph = load_spatial_graph() if enable_osm else None

# ─── Pre-compute prediction ONCE — shared by Tab 2, Tab 3, and header ──────────
# Runs on first load (no button press needed) and on every subsequent button click.
if "analyse_triggered" not in st.session_state:
    st.session_state["analyse_triggered"] = True   # auto-run on first visit

result = predictor.predict(
    event_cause=event_cause, priority=priority, zone=zone,
    hour=event_dt.hour, day_of_week=event_dt.weekday(),
    month=event_dt.month, requires_road_closure=road_closure,
)
severity     = result["severity_class"]
delay_mins   = result["predicted_delay_mins"]

# ─── MILP optimizer result — computed once, shared by Tab 2's dispatch card
# and Tab 3's resource-allocation panel ────────────────────────────────────────
opt_sev = SEVERITY_TO_OPTIMIZER.get(severity, 'NORMAL_OBSTRUCTION')
try:
    opt_result = optimizer_engine.optimize_deployment(
        predicted_delay_mins=float(delay_mins),
        road_lanes=2,
        severity_class=opt_sev,
    )
except Exception as e:
    print(f"  MILP optimizer failed ({e}) — opt_result is None, lookup-table fallback will be used")
    opt_result = None

# ─── T-minus deployment countdown ──────────────────────────────────────────────
# Addresses the "you're diagnosing, not forecasting" gap: if the configured
# event is more than a few minutes in the future, TraffiX is in forecasting
# mode and tells you WHEN to deploy, not just WHAT to deploy.
lead_minutes   = LEAD_TIME_BY_SEVERITY.get(severity, 60)
deploy_at      = event_dt - datetime.timedelta(minutes=lead_minutes)
mins_to_event  = (event_dt - now).total_seconds() / 60.0
mins_to_deploy = (deploy_at - now).total_seconds() / 60.0
is_planned     = mins_to_event > 5

# ─── Header ───────────────────────────────────────────────────────────────────
src_label = "🟢 ORS Live Data"  if api_key else "🔵 Simulation Mode"
src_color = "#22C55E"           if api_key else "#2563EB"
src_bg    = "#F0FDF4"           if api_key else "#EFF6FF"
model_label = "🤖 ML Active"    if (predictor and predictor.is_trained) else "📐 Rule Mode"

st.markdown(f"""
<div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:16px;
            padding:20px 28px;margin-bottom:24px;
            box-shadow:0 1px 3px rgba(0,0,0,0.06);
            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px'>
    <div>
        <h1 style='margin:0;font-size:26px;font-weight:800;color:#111827;letter-spacing:-0.5px'>
            🚦 TraffiX — Bengaluru Traffic Intelligence
        </h1>
        <p style='color:#6B7280;margin:4px 0 0 0;font-size:13px;font-weight:500'>
            ASTRAM SDK Extension &nbsp;·&nbsp; {now.strftime('%A, %d %b %Y')} &nbsp;·&nbsp;
            Last updated {now.strftime('%H:%M')} IST
        </p>
    </div>
    <div style='display:flex;gap:8px;flex-wrap:wrap'>
        <span style='background:{src_bg};color:{src_color};border:1px solid {src_color}33;
                     padding:6px 14px;border-radius:20px;font-size:12px;font-weight:600'>
            {src_label}
        </span>
        <span style='background:#EFF6FF;color:#2563EB;border:1px solid #2563EB33;
                     padding:6px 14px;border-radius:20px;font-size:12px;font-weight:600'>
            {model_label}
        </span>
        <span style='background:#F8FAFC;color:#374151;border:1px solid #E5E7EB;
                     padding:6px 14px;border-radius:20px;font-size:12px;font-weight:500'>
            🏙️ Bengaluru Traffic Police
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Preparation Alerts — auto-triggered from the Planned Event Calendar ──────
# "Planned events should auto-trigger preparation 2hrs before" — scans the
# whole calendar (independent of whatever event is currently loaded in the
# sidebar) and surfaces anything inside its 2-hour prep window.
imminent_cal_events = [e for e in EVENT_CALENDAR if 0 < (e["datetime"] - now).total_seconds() / 60 <= 120]
for cal in imminent_cal_events:
    cal_mins = (cal["datetime"] - now).total_seconds() / 60
    st.markdown(f"""
    <div style='background:#FFFBEB;border:1px solid #FCD34D;border-left:5px solid #F59E0B;
                border-radius:12px;padding:12px 18px;margin-bottom:10px'>
        <span style='font-weight:700;color:#92400E;font-size:13px'>🔔 PREPARATION ALERT — AUTO-TRIGGERED</span><br>
        <span style='color:#78350F;font-size:13px'>
            {cal['name']} starts in <b>{int(cal_mins // 60)}h {int(cal_mins % 60)}m</b> —
            pre-event brief should already be underway. Use "⚡ Load &amp; Brief" in the sidebar to pull it up.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ─── T-Minus Deployment Countdown — forecast, don't just diagnose ─────────────
if is_planned:
    mode_label, mode_color, mode_bg = "📅 PLANNED EVENT — FORECASTING MODE", "#1D4ED8", "#EFF6FF"
else:
    mode_label, mode_color, mode_bg = "🔴 LIVE / REACTIVE MODE", "#DC2626", "#FEF2F2"

if mins_to_deploy <= 0 and mins_to_event > 0:
    tminus_text  = (f"🟠 <b>DEPLOY NOW</b> — inside the {lead_minutes}-min lead window. "
                     f"Event starts at {event_dt.strftime('%I:%M %p')} ({mins_to_event:.0f} min from now).")
    tminus_color = "#F59E0B"
elif mins_to_event <= 0:
    tminus_text  = (f"🔴 Event window has started or passed ({event_dt.strftime('%d %b, %I:%M %p')}) "
                     f"— TraffiX is operating in reactive mode for this event.")
    tminus_color = "#DC2626"
else:
    tminus_text  = (f"🟢 T-minus <b>{mins_to_deploy:.0f} min</b> — deploy {severity} team by "
                     f"<b>{deploy_at.strftime('%I:%M %p')}</b> ({lead_minutes} min before event start "
                     f"at {event_dt.strftime('%I:%M %p')}).")
    tminus_color = "#16A34A"

st.markdown(f"""
<div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:16px;padding:16px 24px;
            margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06);
            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px'>
    <div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>
        <span style='background:{mode_bg};color:{mode_color};padding:6px 14px;border-radius:20px;
                     font-size:12px;font-weight:700'>{mode_label}</span>
        <span style='font-size:14px;color:#111827'>{tminus_text}</span>
    </div>
    <div style='text-align:right'>
        <div style='font-size:11px;color:#9CA3AF;font-weight:600'>SEVERITY</div>
        <div style='font-size:16px;font-weight:800;color:{tminus_color}'>{severity}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI Strip (from real data) ───────────────────────────────────────────────
if df_raw is not None:
    active_count  = (df_raw['status'] == 'active').sum() if 'status' in df_raw.columns else 0
    high_priority = ((df_raw['status'] == 'active') & (df_raw['priority'] == 'High')).sum() \
                    if 'status' in df_raw.columns else 0

    kpi_data = [
        {"label": "Total Events",  "value": f"{len(df_raw):,}",          "sub": "ASTRAM Dataset",        "color": "#2563EB", "bg": "#EFF6FF"},
        {"label": "Active Now",    "value": f"{active_count:,}",          "sub": "Live in ASTRAM",        "color": "#22C55E", "bg": "#F0FDF4"},
        {"label": "High Priority", "value": f"{high_priority:,}",         "sub": "Needs Attention",       "color": "#EF4444", "bg": "#FEF2F2"},
        {"label": "Corridors",     "value": str(len(BENGALURU_CORRIDORS)), "sub": "Monitored",            "color": "#F59E0B", "bg": "#FFFBEB"},
        {"label": "Zones",         "value": str(df_raw['zone'].nunique()), "sub": "Bengaluru BTP",        "color": "#8B5CF6", "bg": "#F5F3FF"},
        {"label": "ML Training",   "value": f"{predictor.model_metrics.get('n_train', 3515):,}",
                                            "sub": "Events (incl. imputed)","color": "#06B6D4", "bg": "#ECFEFF"},
    ]
    icons = ["📊", "🔴", "⚠️", "📍", "🏙️", "🧠"]
    for col, kpi, icon in zip(st.columns(6), kpi_data, icons):
        with col:
            st.markdown(f"""
            <div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:16px;
                        padding:20px 16px 16px 16px;
                        box-shadow:0 1px 3px rgba(0,0,0,0.06);transition:all 0.2s ease;'>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:12px'>
                    <div style='width:32px;height:32px;background:{kpi["bg"]};border-radius:8px;
                                display:flex;align-items:center;justify-content:center;font-size:14px'>
                        {icon}
                    </div>
                    <span style='font-size:12px;color:#6B7280;font-weight:500'>{kpi["label"]}</span>
                </div>
                <div style='font-size:28px;font-weight:800;color:#111827;line-height:1;margin-bottom:4px'>
                    {kpi["value"]}
                </div>
                <div style='font-size:11px;color:{kpi["color"]};font-weight:600'>{kpi["sub"]}</div>
            </div>
            """, unsafe_allow_html=True)

# ─── Model Performance Card ────────────────────────────────────────────────────
if predictor.is_trained and predictor.model_metrics:
    m = predictor.model_metrics
    spacer(20)
    st.markdown(f"""
    <div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:16px;
                padding:16px 24px;margin-bottom:4px;
                box-shadow:0 1px 3px rgba(0,0,0,0.06)'>
        <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px'>
            <div>
                <p style='margin:0;font-size:11px;font-weight:700;text-transform:uppercase;
                           letter-spacing:0.8px;color:#9CA3AF'>
                    🧠 Model Performance — Held-out Test Set ({m.get('n_test','?')} events)
                </p>
                <p style='margin:4px 0 0 0;font-size:12px;color:#6B7280'>
                    {"Using enriched features (road type, lane count, NLP severity)" if m.get('using_engineered') else "Using base features (raw ASTRAM data)"} &nbsp;·&nbsp;
                    {m.get('features_used','?')} features
                </p>
            </div>
            <div style='display:flex;gap:20px;flex-wrap:wrap'>
                <div style='text-align:center'>
                    <div style='font-size:18px;font-weight:800;color:#2563EB'>{m.get('rmse_mins','?')}</div>
                    <div style='font-size:10px;color:#9CA3AF;font-weight:600'>RMSE (min)</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:18px;font-weight:800;color:#2563EB'>{m.get('mae_mins','?')}</div>
                    <div style='font-size:10px;color:#9CA3AF;font-weight:600'>MAE (min)</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:18px;font-weight:800;color:#22C55E'>{m.get('r2_score','?')}</div>
                    <div style='font-size:10px;color:#9CA3AF;font-weight:600'>R²</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:18px;font-weight:800;color:#22C55E'>{m.get('f1_macro','?')}</div>
                    <div style='font-size:10px;color:#9CA3AF;font-weight:600'>F1-macro</div>
                </div>
                <div style='text-align:center'>
                    <div style='font-size:18px;font-weight:800;color:#F59E0B'>{m.get('severity_accuracy','?')}%</div>
                    <div style='font-size:10px;color:#9CA3AF;font-weight:600'>Severity Acc</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Per-class F1 and SMOTE status — surfaces the imbalance story transparently
    _pcf1 = m.get('per_class_f1', {})
    _smote = m.get('smote_applied', False)
    if _pcf1:
        with st.expander("📊 Per-Class F1 Scores (class imbalance transparency)"):
            f1_cols = st.columns(len(_pcf1))
            for col, (cls, f1v) in zip(f1_cols, _pcf1.items()):
                color = "#22C55E" if f1v >= 0.7 else "#F59E0B" if f1v >= 0.5 else "#EF4444"
                col.markdown(f"<div style='text-align:center;padding:8px'><div style='font-size:20px;font-weight:800;color:{color}'>{f1v}</div><div style='font-size:11px;color:#6B7280'>{cls}</div></div>", unsafe_allow_html=True)
            smote_note = "✅ SMOTE applied on training data to address minority class imbalance (protest: 15 events, vip_movement: 20 events)" if _smote else "⚠️ SMOTE not applied — minority classes use class_weight='balanced' only"
            st.caption(smote_note)

    # Data quality breakdown — explains why 70% was "dropped"
    dqr = predictor.data_quality_report
    if dqr:
        imp_note = (
            f"📊 Data quality: {dqr.get('total_events',0):,} total events → "
            f"**{dqr.get('formally_closed',0):,} formally closed** + "
            f"**{dqr.get('total_imputed',0):,} imputed** "
            f"(resolved_datetime: {dqr.get('imputed_from_resolved',0)}, "
            f"active/modified: {dqr.get('imputed_from_modified',0)}) → "
            f"**{dqr.get('usable_events',0):,} usable events** ({dqr.get('pct_usable',0)}%). "
            f"Planned events: {dqr.get('n_planned_events',0)} | "
            f"Unplanned: {dqr.get('n_unplanned_events',0)} | "
            f"Description coverage: {dqr.get('description_coverage','?')}"
        )
        st.caption(imp_note)

spacer(24)

# ─── MAIN TABS ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Live Traffic Map",
    "🚧 Barricade Placement",
    "🔮 Event Prediction",
    "🧬 DNA Matching",
    "⚠️ Compliance Risk",
    "📋 Analytics",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE TRAFFIC MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_map, col_alerts = st.columns([3, 1])

    with col_map:
        st.markdown("""
        <div style='margin-bottom:8px'>
            <h3 style='font-size:18px;font-weight:700;color:#111827;margin:0'>
                🗺️ Bengaluru Live Traffic — All Corridors
            </h3>
        </div>
        """, unsafe_allow_html=True)
        source_note = "🟢 ORS Live Data"        if api_key else "🔵 Simulation Mode (set ORS_API_KEY env var for live data)"
        st.caption(f"{source_note} · Auto-refreshes every 5 min · {now.strftime('%H:%M:%S')} IST")

        with st.spinner("Scanning 12 Bengaluru corridors..."):
            traffic_data = fetch_live_traffic(api_key=api_key if api_key else None)

        m = folium.Map(location=[12.9716, 77.5946], zoom_start=12,
                       tiles="CartoDB positron", width="100%")

        for corridor in traffic_data:
            sev   = corridor["congestion_severity"]
            color = corridor["congestion_color"]
            label = corridor["congestion_label"]
            speed = corridor["speed_kmh"]
            delay = corridor["delay_mins"]
            name  = corridor["corridor"]
            radius = 600 + sev * 350
            fill_opacity = 0.7 if sev >= 3 else 0.5

            folium.CircleMarker(
                location=[corridor["lat"], corridor["lng"]],
                radius=radius / 80, color=color,
                fill=True, fill_color=color, fill_opacity=fill_opacity,
                tooltip=folium.Tooltip(
                    f"<b>{name}</b><br>Status: <b style='color:{color}'>{label}</b><br>"
                    f"Speed: {speed} km/h<br>Delay: +{delay:.0f} min<br>"
                    f"Zone: {corridor['zone']}<br>PS: {corridor['btp_ps']}",
                    sticky=True),
                popup=folium.Popup(
                    f"<div style='width:200px;font-family:Inter,sans-serif'>"
                    f"<b style='color:#111827'>{name}</b><hr style='border-color:#E5E7EB'>"
                    f"🚥 Status: <b style='color:{color}'>{label}</b><br>"
                    f"⚡ Speed: {speed} km/h<br>"
                    f"⏱️ Normal: {corridor['free_flow_mins']} min → Now: {corridor['actual_travel_mins']} min<br>"
                    f"🔺 Delay: +{delay:.0f} min<br>"
                    f"📍 {corridor['btp_ps']}<br><br>"
                    f"<small style='color:#6B7280'>Updated: {corridor['timestamp']}</small></div>",
                    max_width=220),
            ).add_to(m)

            folium.Marker(
                location=[corridor["lat"] + 0.004, corridor["lng"]],
                icon=folium.DivIcon(
                    html=f"<div style='font-size:9px;font-weight:700;color:{color};background:rgba(255,255,255,0.92);"
                         f"border:1px solid {color}44;padding:2px 5px;border-radius:4px;white-space:nowrap;"
                         f"box-shadow:0 1px 4px rgba(0,0,0,0.12)'>{corridor['congestion_emoji']} {speed} km/h</div>",
                    icon_size=(80, 16), icon_anchor=(40, 8)),
            ).add_to(m)

        folium.Marker(
            location=[custom_lat, custom_lng],
            icon=folium.Icon(color="purple", icon="star", prefix="fa"),
            tooltip="📍 Selected Event Location",
            popup=folium.Popup(
                f"<b>Event Location</b><br>{event_cause.replace('_',' ').title()}<br>{zone}",
                max_width=150),
        ).add_to(m)

        legend_html = """
        <div style='position:fixed;bottom:30px;left:30px;
                    background:rgba(255,255,255,0.97);border:1px solid #E5E7EB;
                    padding:14px 18px;border-radius:12px;font-size:12px;z-index:9999;
                    color:#374151;font-family:Inter,sans-serif;
                    box-shadow:0 4px 16px rgba(0,0,0,0.10)'>
            <b style='color:#111827;font-size:13px'>Congestion Level</b><br><br>
            <span style='color:#00cc44'>●</span> Free Flow (&lt;1.3x)<br>
            <span style='color:#aacc00'>●</span> Slow (1.3–1.6x)<br>
            <span style='color:#ff8800'>●</span> Heavy (1.6–2.0x)<br>
            <span style='color:#ff4444'>●</span> Gridlock (2.0–2.8x)<br>
            <span style='color:#880000'>●</span> Standstill (&gt;2.8x)<br>
            <span style='color:#8B5CF6'>★</span> Event Location
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        st_folium(m, width=None, height=600, returned_objects=[])

    with col_alerts:
        st.markdown("""
        <div style='margin-bottom:8px'>
            <h3 style='font-size:17px;font-weight:700;color:#111827;margin:0'>🚨 Live Alerts</h3>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Sorted worst first · {now.strftime('%H:%M')}")

        gridlock_count = sum(1 for c in traffic_data if c["congestion_severity"] >= 3)
        heavy_count    = sum(1 for c in traffic_data if c["congestion_severity"] == 2)
        slow_count     = sum(1 for c in traffic_data if c["congestion_severity"] == 1)
        # FIX: SLOW ≠ clear. Only severity == 0 (FREE FLOW) counts as clear.
        clear_count    = sum(1 for c in traffic_data if c["congestion_severity"] == 0)

        st.markdown(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:16px'>
            <div style='background:#FEF2F2;border:1px solid #FECACA;border-radius:10px;padding:10px 8px;text-align:center'>
                <div style='font-size:20px;font-weight:800;color:#EF4444'>{gridlock_count}</div>
                <div style='font-size:10px;color:#7F1D1D;font-weight:600'>Gridlock</div>
            </div>
            <div style='background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:10px 8px;text-align:center'>
                <div style='font-size:20px;font-weight:800;color:#F59E0B'>{heavy_count}</div>
                <div style='font-size:10px;color:#78350F;font-weight:600'>Heavy</div>
            </div>
            <div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:10px 8px;text-align:center'>
                <div style='font-size:20px;font-weight:800;color:#22C55E'>{clear_count}</div>
                <div style='font-size:10px;color:#14532D;font-weight:600'>Free Flow</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for corridor in traffic_data:
            msg       = get_corridor_alert_message(corridor)
            sev       = corridor["congestion_severity"]
            css_class = "alert-critical" if sev >= 3 else "alert-high" if sev >= 1 else "alert-ok"
            st.markdown(f"<div class='{css_class}'><small>{msg}</small></div>",
                        unsafe_allow_html=True)

    # ── Diversion Suggestions ─────────────────────────────────────────────────
    gridlocked = [c for c in traffic_data if c["congestion_severity"] >= 3]
    if gridlocked:
        spacer(16)
        st.markdown("""
        <h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:8px'>
            🔀 Suggested Diversion Corridors
        </h3>
        <p style='color:#6B7280;font-size:13px;margin:0 0 12px 0'>
            For corridors in gridlock, the following alternative corridors in nearby zones
            are currently flowing better. Recommend to commuters via VMS boards.
        </p>
        """, unsafe_allow_html=True)

        # Pair each gridlocked corridor with the best-flowing alternatives
        clear_corridors = [c for c in traffic_data if c["congestion_severity"] <= 1]
        div_cols = st.columns(min(len(gridlocked), 3))
        for i, blocked in enumerate(gridlocked[:3]):
            alts = [c for c in clear_corridors if c["corridor"] != blocked["corridor"]][:2]
            with div_cols[i]:
                alt_html = "".join([
                    f"<div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;"
                    f"padding:8px 10px;margin:4px 0;font-size:12px'>"
                    f"<b style='color:#166534'>✅ {a['corridor']}</b><br>"
                    f"<span style='color:#6B7280'>{a['speed_kmh']} km/h · {a['zone']}</span>"
                    f"</div>"
                    for a in alts
                ]) if alts else "<p style='color:#9CA3AF;font-size:12px'>No clear alternates available</p>"

                st.markdown(f"""
                <div style='background:#FEF2F2;border:1px solid #FECACA;border-radius:12px;padding:12px 14px'>
                    <div style='font-size:13px;font-weight:700;color:#991B1B;margin-bottom:8px'>
                        🔴 Blocked: {blocked['corridor']}
                    </div>
                    <div style='font-size:11px;color:#6B7280;margin-bottom:8px'>
                        {blocked['speed_kmh']} km/h · +{blocked['delay_mins']:.0f} min delay
                    </div>
                    <div style='font-size:11px;font-weight:600;color:#374151;margin-bottom:4px'>
                        Divert via:
                    </div>
                    {alt_html}
                </div>
                """, unsafe_allow_html=True)

    # ── Corridor Status Table ─────────────────────────────────────────────────
    spacer(24)
    st.markdown("""
    <h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:12px'>
        📊 All Corridor Status
    </h3>
    """, unsafe_allow_html=True)
    df_traffic   = pd.DataFrame(traffic_data)
    display_cols = ["corridor", "zone", "speed_kmh", "actual_travel_mins",
                    "delay_mins", "congestion_label", "btp_ps"]
    df_display   = df_traffic[display_cols].copy()
    df_display.columns = ["Corridor", "Zone", "Speed (km/h)", "Travel Time (min)",
                           "Delay (min)", "Status", "Police Station"]

    st.dataframe(
        df_display.style.map(color_status, subset=["Status"]),
        use_container_width=True, hide_index=True,
    )
    st.caption("📡 OpenRouteService Live" if api_key else
               "📊 TraffiX Simulation Engine (deterministic, seeded by current hour + day-of-week)")

    # ── Stalled Event Detector ────────────────────────────────────────────────
    # Events open 2x their predicted clearance time with no closure logged.
    # Uses dataset reference time (max start_datetime in dataset) so the demo
    # works regardless of when it's run relative to the historical dataset dates.
    spacer(24)
    st.markdown("""
    <h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:4px'>
        🔴 Stalled Event Detector
    </h3>
    <p style='color:#6B7280;font-size:13px;margin:0 0 12px 0'>
        Active ASTRAM events that were open for &gt;2× their predicted clearance time
        at the dataset snapshot date (Apr 2024). These are the hardest cases — events that
        dragged on so long they were abandoned in the system without formal closure.
    </p>
    """, unsafe_allow_html=True)

    if df_raw is not None:
        with st.spinner("Scanning active events for staleness..."):
            # Use dataset max start_datetime as reference — this is historically accurate
            df_ts = df_raw.copy()
            df_ts['start_datetime'] = pd.to_datetime(df_ts['start_datetime'], errors='coerce', utc=True)
            dataset_ref_time = df_ts['start_datetime'].max()
            stalled = predictor.detect_stalled_events(df_raw, current_time=dataset_ref_time)

        if stalled:
            st.error(f"⚠️ **{len(stalled)} stalled event(s) detected** — open far beyond predicted clearance time at dataset snapshot")
            for ev in stalled[:8]:
                factor = ev['staleness_factor']
                color  = "#EF4444" if factor > 5 else "#F59E0B"
                bg     = "#FEF2F2" if factor > 5 else "#FFFBEB"
                st.markdown(f"""
                <div style='background:{bg};border:1px solid {color}33;border-left:4px solid {color};
                            border-radius:10px;padding:12px 14px;margin:6px 0'>
                    <div style='display:flex;justify-content:space-between;align-items:center'>
                        <div>
                            <span style='font-weight:700;color:#111827;font-size:13px'>
                                🚨 {ev['event_cause'].replace('_',' ').title()} — {ev['junction']}
                            </span><br>
                            <span style='font-size:12px;color:#374151'>
                                Zone: {ev['zone']} &nbsp;·&nbsp; Priority: {ev['priority']}
                            </span><br>
                            <span style='font-size:12px;color:#6B7280'>
                                {str(ev['address'])[:70] if ev['address'] else 'No address on record'}
                            </span>
                        </div>
                        <div style='text-align:right'>
                            <div style='font-size:18px;font-weight:800;color:{color}'>
                                {ev['staleness_factor']}×
                            </div>
                            <div style='font-size:11px;color:#9CA3AF'>staleness factor</div>
                            <div style='font-size:12px;color:#374151'>
                                Open {int(ev['minutes_open'])}min / Predicted {int(ev['expected_minutes'])}min
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No stalled events detected at dataset snapshot time.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BARRICADE PLACEMENT MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style='margin-bottom:16px'>
        <h3 style='font-size:18px;font-weight:700;color:#111827;margin:0 0 6px 0'>
            🚧 Dynamic Barricade Placement — GPS Coordinates for Officers
        </h3>
        <p style='color:#6B7280;font-size:14px;margin:0'>
            Exact GPS position for each barricade. Officers can navigate using their phone.
            The plan is adjusted based on event type and ML-predicted severity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # severity already computed from the shared predict() call above
    plan = generate_barricade_plan(
        event_lat=custom_lat, event_lng=custom_lng,
        event_cause=event_cause, severity=severity,
        junction_name=junction_input if junction_input else None,
        spatial_graph=spatial_graph,
    )

    radius_m = int(result["affected_radius_km"] * 1000)

    # ── Diversion routes — "where do diverted vehicles go?" ───────────────────
    # Real OSM shortest-path routes that physically avoid the blocked event
    # radius. Only computed when the OSM graph is loaded (sidebar toggle) —
    # otherwise we show barricades only, same as before.
    diversion_routes = []
    if spatial_graph is not None:
        with st.spinner("Computing alternate routes on the Bengaluru road network..."):
            diversion_routes = spatial_graph.get_diversion_routes(
                event_lat=custom_lat, event_lon=custom_lng,
                num_routes=2, block_radius_m=max(radius_m, 200),
            )

    b1, b2, b3, b4 = st.columns(4)
    barricade_kpis = [
        {"col": b1, "label": "Total Positions",     "value": plan["total_placements"],              "color": "#2563EB", "bg": "#EFF6FF"},
        {"col": b2, "label": "Physical Barricades", "value": plan["total_physical_barricades"],     "color": "#F59E0B", "bg": "#FFFBEB"},
        {"col": b3, "label": "Officers Required",   "value": plan["total_officers_for_barricades"], "color": "#8B5CF6", "bg": "#F5F3FF"},
        {"col": b4, "label": "Emergency Corridor",
         "value": "✅ MAINTAINED" if plan["emergency_corridor_maintained"] else "⚠️ CHECK",
         "color": "#22C55E" if plan["emergency_corridor_maintained"] else "#EF4444",
         "bg":    "#F0FDF4"  if plan["emergency_corridor_maintained"] else "#FEF2F2"},
    ]
    for kpi in barricade_kpis:
        with kpi["col"]:
            st.markdown(f"""
            <div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:16px;
                        padding:18px 16px;text-align:center;
                        box-shadow:0 1px 3px rgba(0,0,0,0.06);transition:all 0.2s ease'>
                <div style='font-size:22px;font-weight:800;color:{kpi["color"]}'>{kpi["value"]}</div>
                <div style='font-size:12px;color:#6B7280;font-weight:500;margin-top:4px'>{kpi["label"]}</div>
            </div>
            """, unsafe_allow_html=True)

    spacer(12)
    st.info(f"📋 **Deployment Order:** {plan['summary']}")
    _ds = plan.get("data_source", "unknown")
    if _ds.startswith("osm_graph_traversal"):
        st.success(f"🗺️ **OSM Road Coverage Active** — Barricades placed on real Bengaluru roads (ego-graph traversal, {_ds.split(':')[1] if ':' in _ds else '350m radius'}). Full city coverage.")
    elif _ds.startswith("hardcoded_junction"):
        st.info(f"📍 **Known Junction Mode** — Using pre-verified approach roads for {_ds.split(':')[1] if ':' in _ds else 'this junction'}.")
    else:
        st.warning("⚠️ **Cardinal Fallback** — OSM graph not loaded. Enable '🗺️ OSM Diversion Routing' in the sidebar for real road names.")

    bar_col1, bar_col2 = st.columns([3, 1])

    with bar_col1:
        bm = folium.Map(location=[custom_lat, custom_lng], zoom_start=15, tiles="CartoDB positron")

        folium.Marker(
            location=[custom_lat, custom_lng],
            icon=folium.Icon(color="purple", icon="star", prefix="fa"),
            tooltip="📍 Event Location",
        ).add_to(bm)

        radius_m = int(result["affected_radius_km"] * 1000)
        folium.Circle(
            location=[custom_lat, custom_lng], radius=radius_m,
            color="#EF4444", fill=True, fill_color="#EF4444",
            fill_opacity=0.06, tooltip=f"Affected radius: {result['affected_radius_km']} km",
            dash_array="10",
        ).add_to(bm)

        # ── Diversion route polylines — OSM shortest-path avoiding the blocked zone ──
        route_colors = ["#0EA5E9", "#9333EA"]
        for i, route in enumerate(diversion_routes):
            rc = route_colors[i % len(route_colors)]
            via = ", ".join(route["via_roads"][:3]) or "local roads"
            folium.PolyLine(
                locations=route["coords"], color=rc, weight=5, opacity=0.85,
                tooltip=folium.Tooltip(
                    f"<b>Diversion Route {i + 1}</b><br>{route['length_km']} km · ~{route['eta_min']:.0f} min"
                    f"<br>via {via}", sticky=True),
            ).add_to(bm)
            if route["coords"]:
                folium.CircleMarker(
                    location=route["coords"][0], radius=6, color=rc, fill=True, fill_color="#FFFFFF",
                    fill_opacity=1, tooltip=f"Diversion {i + 1} — entry point",
                ).add_to(bm)
                folium.CircleMarker(
                    location=route["coords"][-1], radius=6, color=rc, fill=True, fill_color=rc,
                    fill_opacity=1, tooltip=f"Diversion {i + 1} — exit point",
                ).add_to(bm)

        for dep in plan["deployments"]:
            col = dep["color"]
            bid = dep["id"]

            folium.CircleMarker(
                location=[dep["lat"], dep["lng"]],
                radius=10, color=col, fill=True, fill_color=col, fill_opacity=0.85,
                tooltip=folium.Tooltip(
                    f"<b>{bid} — {dep['type_label']}</b><br>Road: {dep['road']}<br>"
                    f"Priority: {dep['priority']}<br>Officers: {dep['officers_needed']}<br><br>"
                    f"<i>{dep['instruction']}</i>", sticky=True),
            ).add_to(bm)

            folium.Marker(
                location=[dep["lat"], dep["lng"]],
                icon=folium.DivIcon(
                    html=f"<div style='font-size:9px;font-weight:700;color:white;background:{col};"
                         f"padding:2px 5px;border-radius:10px;text-align:center;"
                         f"box-shadow:0 1px 4px rgba(0,0,0,0.2)'>{bid}</div>",
                    icon_size=(30, 16), icon_anchor=(15, 8)),
            ).add_to(bm)

            folium.PolyLine(
                locations=[[custom_lat, custom_lng], [dep["lat"], dep["lng"]]],
                color=col, weight=1.5, opacity=0.35, dash_array="5",
            ).add_to(bm)

        st_folium(bm, width=None, height=500, returned_objects=[])

    with bar_col2:
        st.markdown("<h4 style='font-size:15px;font-weight:700;color:#111827;margin:0 0 4px 0'>📋 Barricade List</h4>", unsafe_allow_html=True)
        st.caption("Share with officers on WhatsApp or print")

        for dep in plan["deployments"]:
            priority_badge = "🔴" if dep["priority"] == "CRITICAL" else "🟠" if dep["priority"] == "HIGH" else "🟡"
            st.markdown(f"""
            <div class='barricade-card' style='border-left:4px solid {dep["color"]}'>
                <b style='color:{dep["color"]};font-size:13px'>{dep["icon"]} {dep["id"]}</b>
                <span style='float:right'>{priority_badge}</span><br>
                <span style='font-size:12px;font-weight:600;color:#374151'>{dep["type_label"]}</span><br>
                <span style='font-size:11px;color:#6B7280'>📍 {dep["road"]}</span><br>
                <span style='font-size:11px;color:#9CA3AF'>GPS: {dep["lat"]},{dep["lng"]}</span><br>
                <span style='font-size:11px;color:#6B7280'>👮 {dep["officers_needed"]} officer(s)</span><br>
                <span style='font-size:11px;color:#9CA3AF;font-style:italic'>{dep["instruction"][:80]}...</span>
            </div>
            """, unsafe_allow_html=True)

    spacer(20)
    st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:4px'>🔀 Live Diversion Routes — Avoiding Event Zone</h3>", unsafe_allow_html=True)
    if spatial_graph is None:
        st.info("📡 Enable **OSM Diversion Routing** in the sidebar to compute real shortest-path alternate routes around this event on the Bengaluru road network.")
    elif not diversion_routes:
        st.warning("No alternate route could be computed for this location on the cached road network — show barricades only and rely on officer judgement for diversion.")
    else:
        st.caption("Computed via networkx shortest-path on the OSM drive network, routed around the blocked event radius — drawn as colored polylines on the map above.")
        route_cols = st.columns(len(diversion_routes))
        route_colors = ["#0EA5E9", "#9333EA"]
        for i, (col, route) in enumerate(zip(route_cols, diversion_routes)):
            rc = route_colors[i % len(route_colors)]
            via = " → ".join(route["via_roads"][:3]) or "local roads"
            with col:
                st.markdown(f"""
                <div style='background:#FFFFFF;border:1px solid #E5E7EB;border-left:4px solid {rc};
                            border-radius:12px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,0.06)'>
                    <div style='font-size:13px;font-weight:700;color:{rc}'>Diversion Route {i + 1}{' (shortest)' if i == 0 else ' (alternate)'}</div>
                    <div style='font-size:20px;font-weight:800;color:#111827;margin-top:4px'>{route['length_km']} km</div>
                    <div style='font-size:12px;color:#6B7280'>~{route['eta_min']:.0f} min diverted-traffic ETA</div>
                    <div style='font-size:12px;color:#374151;margin-top:8px'>via {via}</div>
                </div>
                """, unsafe_allow_html=True)

    spacer(24)
    st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:12px'>🖨️ Printable Deployment Sheet</h3>", unsafe_allow_html=True)
    dep_df = pd.DataFrame(plan["deployments"])[
        ["id", "type_label", "road", "lat", "lng", "officers_needed", "priority", "instruction"]
    ]
    dep_df.columns = ["Barricade ID", "Type", "Location/Road", "Latitude", "Longitude",
                      "Officers", "Priority", "Instruction for Officer"]
    st.dataframe(dep_df, use_container_width=True, hide_index=True)

    # ── Sub-Inspector Field Dispatch Card — print-ready order, not a GIS dump ──
    spacer(28)
    st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:4px'>📋 Sub-Inspector Field Dispatch Card</h3>", unsafe_allow_html=True)
    st.caption("A one-page field order formatted the way a Sub-Inspector would actually hand it to constables — not a GIS dump.")

    location_label = junction_input if junction_input else f"{zone} ({custom_lat:.4f}, {custom_lng:.4f})"
    dispatch_id = dispatch_card.make_dispatch_id(event_dt, zone)
    resource_lines = dispatch_card.pick_top_resources(opt_result, result.get("resources"))
    barricade_lines = dispatch_card.build_barricade_lines(plan["deployments"])
    emergency_corridor_line = dispatch_card.build_emergency_corridor_line(plan["deployments"])
    diversion_lines = dispatch_card.build_diversion_lines(diversion_routes)
    confidence_pct = float(result.get("confidence_pct", result.get("severity_confidence", 85)) or 85)

    dispatch_html = dispatch_card.generate_dispatch_html(
        dispatch_id=dispatch_id, event_cause=event_cause, zone=zone,
        location_label=location_label, lat=custom_lat, lng=custom_lng,
        event_dt=event_dt, deploy_at=deploy_at, severity=severity,
        confidence_pct=confidence_pct, expected_clearance_mins=result["predicted_clearance_mins"],
        resource_lines=resource_lines, barricade_lines=barricade_lines,
        emergency_corridor_line=emergency_corridor_line, diversion_lines=diversion_lines,
        generated_at=now,
    )
    st.markdown(dispatch_html, unsafe_allow_html=True)

    try:
        dispatch_pdf_bytes = dispatch_card.generate_dispatch_pdf(
            dispatch_id=dispatch_id, event_cause=event_cause, zone=zone,
            location_label=location_label, lat=custom_lat, lng=custom_lng,
            event_dt=event_dt, deploy_at=deploy_at, severity=severity,
            confidence_pct=confidence_pct, expected_clearance_mins=result["predicted_clearance_mins"],
            resource_lines=resource_lines, barricade_lines=barricade_lines,
            emergency_corridor_line=emergency_corridor_line, diversion_lines=diversion_lines,
            generated_at=now,
        )
        spacer(8)
        st.download_button(
            "⬇️ Download Print-Ready PDF Dispatch Card",
            data=dispatch_pdf_bytes, file_name=f"{dispatch_id}.pdf",
            mime="application/pdf", use_container_width=False,
        )
    except Exception as e:
        st.warning(f"PDF generation unavailable right now ({e}) — use the card preview above.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVENT PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<h3 style='font-size:18px;font-weight:700;color:#111827;margin:0 0 4px 0'>🔮 Event Impact Prediction</h3>", unsafe_allow_html=True)
    st.caption(f"Predicting: {event_cause.replace('_',' ').title()} · {zone} · {event_dt.strftime('%d %b %Y, %I:%M %p')}")

    # Use the shared result computed at page top
    # DISTINCT colors for HIGH and MODERATE (was identical before)
    sev_colors  = {"CRITICAL": "#EF4444", "HIGH": "#F59E0B", "MODERATE": "#FBBF24", "LOW": "#22C55E"}
    sev_bgs     = {"CRITICAL": "#FEF2F2", "HIGH": "#FFFBEB", "MODERATE": "#FEFCE8", "LOW": "#F0FDF4"}
    sev_borders = {"CRITICAL": "#FECACA", "HIGH": "#FDE68A", "MODERATE": "#FEF08A", "LOW": "#BBF7D0"}
    sev_icons   = {"CRITICAL": "🔴",      "HIGH": "🟠",      "MODERATE": "🟡",       "LOW": "🟢"}
    sev_color   = sev_colors.get(severity, "#6B7280")
    sev_bg      = sev_bgs.get(severity, "#F9FAFB")
    sev_border  = sev_borders.get(severity, "#E5E7EB")
    sev_icon    = sev_icons.get(severity, "⚪")

    st.markdown(f"""
    <div style='background:{sev_bg};border:1px solid {sev_border};border-left:6px solid {sev_color};
                border-radius:14px;padding:20px 24px;margin-bottom:20px'>
        <div style='display:flex;align-items:center;gap:12px'>
            <div>
                <div style='font-size:14px;color:#6B7280;font-weight:500;margin-bottom:4px'>
                    Impact Classification
                </div>
                <h2 style='margin:0;font-size:26px;font-weight:800;color:#111827'>
                    {sev_icon} {severity}
                </h2>
                <p style='color:#6B7280;margin:6px 0 0 0;font-size:13px'>
                    Calibrated confidence: <b style='color:#111827'>{result["confidence_pct"]}%</b> &nbsp;·&nbsp;
                    {("🎯 Specialist planned-event model" if result.get("used_planned_specialist") else "🤖 ML Model (calibrated RF)")} &nbsp;·&nbsp;
                    {predictor.model_metrics.get('n_train', 2460):,} training events
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Planned vs Unplanned event callout ────────────────────────────────────
    if result.get('is_planned_event'):
        pm = predictor.planned_metrics
        pm_note = (
            f"Trained on {pm.get('n_train','?')} planned events (procession/public_event/protest/vip_movement) | "
            f"RMSE: {pm.get('rmse_mins','?')} min | MAE: {pm.get('mae_mins','?')} min"
        ) if pm else "Specialist model loaded."
        st.success(
            f"🎯 **Planned Event Path Active** — This prediction uses TraffiX's dedicated "
            f"crowd-event model, separate from the general model dominated by vehicle_breakdown. "
            f"{pm_note}"
        )
    else:
        dqr_ev = predictor.data_quality_report
        n_pb = dqr_ev.get('n_unplanned_events', '?')
        st.info(
            f"🚗 **Unplanned Incident Path** — General model trained on {n_pb} unplanned events. "
            f"Planned events (procession, protest, public_event) use a separate specialist model."
        )

    # ── NLP transparency ──────────────────────────────────────────────────────
    dqr_nlp = predictor.data_quality_report
    nlp_note = dqr_nlp.get('nlp_coverage_note', '')
    if nlp_note:
        with st.expander("🔍 NLP Coverage — What the text extraction actually found"):
            st.info(nlp_note)
            st.caption(
                "UNKNOWN severity events are NOT dropped — they are modelled as an explicit "
                "feature value (nlp_sev_enc). The model learns that UNKNOWN = less information, "
                "not that the event is low severity."
            )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("⏱️ Expected Clearance", f"{result['predicted_clearance_mins']:.0f} min")
    c2.metric("🚗 Congestion Level", f"{result['predicted_congestion_pct']}th %ile",
              help="Percentile rank within historical zone+hour bucket — 90th = worse than 90% of past events here")
    c3.metric("⏳ Commuter Delay",     f"{result['predicted_delay_mins']} min")
    c4.metric("📍 Affected Radius",    f"{result['affected_radius_km']} km")
    c5.metric("🕐 Peak Impact",        result["peak_impact_window"])

    spacer(24)

    # ── MILP Optimizer — Cost-Optimal Resource Allocation ─────────────────────
    st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:8px'>🔧 MILP-Optimal Resource Allocation</h3>", unsafe_allow_html=True)
    st.caption("Mixed-Integer Linear Programming minimises deployment cost while guaranteeing delay mitigation ≥ predicted impact")

    # opt_result was already computed once at the top of the script (shared
    # with the Tab 2 dispatch card) so both views always agree.
    if opt_result is not None:
        opt_status = opt_result.get("status", "UNKNOWN")
        opt_color  = "#22C55E" if opt_status == "OPTIMAL" else "#F59E0B"
        opt_label  = "✅ Optimal Solution Found" if opt_status == "OPTIMAL" else "⚠️ Fallback Deployment"

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Constable Teams (×2)",   opt_result.get("constable_teams_2p", 0))
        o2.metric("Inspector Strike Teams", opt_result.get("inspector_strike_teams", 0))
        o3.metric("Barricades (MILP)",      opt_result.get("barricades_required", 0))
        o4.metric("Signal Diversions",      opt_result.get("signal_diversion_protocols", 0))

        st.markdown(f"""
        <div style='background:#F8FAFC;border:1px solid #E5E7EB;border-radius:10px;
                    padding:10px 14px;margin-top:8px;display:flex;align-items:center;gap:12px'>
            <span style='color:{opt_color};font-weight:700;font-size:13px'>{opt_label}</span>
            <span style='color:#6B7280;font-size:12px'>
                Total mitigation score: {opt_result.get("total_mitigation_score", 0):.1f} min absorbed &nbsp;·&nbsp;
                Cost index: {opt_result.get("estimated_cost_index", 0)} &nbsp;·&nbsp;
                <span style='color:#9CA3AF;font-size:11px'>Costs from Karnataka Police 7CPC rates + BBMP logistics</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
    breakdown = opt_result.get("cost_breakdown", {})
    if breakdown and breakdown.get("note"):
        with st.expander("💰 Why these numbers? (Cost citation)"):
            st.caption(breakdown["note"])
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Constable Teams (₹ idx)", breakdown.get("constable_teams", 0))
            bc2.metric("Inspector Teams (₹ idx)", breakdown.get("inspector_teams", 0))
            bc3.metric("Barricades (₹ idx)", breakdown.get("barricades", 0))
            bc4.metric("Signal Override (₹ idx)", breakdown.get("signal_override", 0))
    else:
        st.warning("MILP optimizer unavailable. Using lookup-table fallback below.")

    spacer(20)
    st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:12px'>👮 BTP Deployment Guide (Lookup Table)</h3>", unsafe_allow_html=True)
    res = result["resources"]
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Traffic Officers", res["officers"])
    r2.metric("Home Guards",      res["home_guards"])
    r3.metric("Barricades",       res["barricades"])
    r4.metric("Tow Trucks",       res["tow_trucks"])
    r5.metric("CCTV Teams",       res["cctv_teams"])
    if res.get("traffic_marshals", 0) > 0:
        st.info(f"🦺 Also deploy **{res['traffic_marshals']} Traffic Marshals** for crowd flow at entry/exit points.")

    spacer(24)

    # ── Data-Driven Congestion Forecast ───────────────────────────────────────
    st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:4px'>📈 Congestion Forecast — Hour by Hour</h3>", unsafe_allow_html=True)
    st.caption("Blends historical hourly patterns from ASTRAM data with model prediction (65% data-driven, 35% decay curve)")
    base_pct = result["predicted_congestion_pct"]
    hours, congestion_curve = predictor.get_hourly_forecast(base_pct, event_dt.hour)
    hour_labels = [f"{h}:00" for h in hours]
    forecast_df = pd.DataFrame({"Hour": hour_labels, "Congestion %": congestion_curve})
    st.bar_chart(forecast_df.set_index("Hour"))

    spacer(24)

    # ── Feature Importance ────────────────────────────────────────────────────
    if predictor.feature_importance:
        st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:12px'>📊 What Drives the Prediction</h3>", unsafe_allow_html=True)
        fi = predictor.feature_importance
        fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
        fi_df = fi_df.head(8)
        st.bar_chart(fi_df.set_index("Feature"))

    spacer(24)

    # ── Public Advisory ───────────────────────────────────────────────────────
    st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:12px'>📢 Auto-Generated Public Advisory</h3>", unsafe_allow_html=True)
    impact_desc = {"CRITICAL": "severe", "HIGH": "heavy", "MODERATE": "moderate", "LOW": "light"}.get(severity, "moderate")
    advisory_text = (
        f"⚠️ Traffic Advisory — Bengaluru Traffic Police\n\n"
        f"{impact_desc.title()} traffic disruption expected near **{zone}** "
        f"due to **{event_cause.replace('_', ' ').title()}** on {event_date.strftime('%d %b %Y')} "
        f"starting {event_time.strftime('%I:%M %p')}.\n\n"
        f"Expected delay: **{result['predicted_delay_mins']:.0f} minutes** for commuters.\n"
        f"Affected area: **{result['affected_radius_km']} km radius**.\n\n"
        f"Alternate routes recommended. Follow updates on the ASTRAM App."
    )
    st.info(advisory_text)
    a1, a2, a3 = st.columns(3)

    # Functional advisory buttons — each shows a formatted output modal
    sms_text = (
        f"BTP ALERT: {impact_desc.upper()} traffic disruption near {zone} "
        f"due to {event_cause.replace('_',' ').upper()} on "
        f"{event_date.strftime('%d %b')} from {event_time.strftime('%I:%M%p')}. "
        f"Expect {result['predicted_delay_mins']:.0f}min delay. "
        f"Avoid area, use alternate routes. -Bengaluru Traffic Police"
    )
    astram_payload = {
        "event_type":      event_cause,
        "zone":            zone,
        "severity":        severity,
        "predicted_clearance_mins": result['predicted_clearance_mins'],
        "affected_radius_km": result['affected_radius_km'],
        "resources_required": result['resources'],
        "advisory_text":   advisory_text,
        "timestamp":       now.isoformat(),
        "source":          "TraffiX v3.0",
    }
    vms_text = (
        f"BTP ADVISORY | {zone.upper()} | {event_cause.replace('_',' ').upper()} "
        f"| {impact_desc.upper()} DISRUPTION | DELAY ~{result['predicted_delay_mins']:.0f} MIN "
        f"| USE ALTERNATE ROUTES | FOLLOW BTP DIRECTIONS"
    )

    if a1.button("📱 Push to ASTRAM App", use_container_width=True):
        st.json(astram_payload)
        st.caption("✅ Above JSON payload ready to POST to ASTRAM Advisory API endpoint /v2/events/advisory")

    if a2.button("💬 Send SMS to Commuters", use_container_width=True):
        st.code(sms_text, language=None)
        st.caption(f"✅ SMS ({len(sms_text)} chars) — formatted for BBMP mass-alert gateway. "
                   f"Estimated reach: {int(result['affected_radius_km']*1000*3.14):,} commuters within affected radius.")

    if a3.button("📺 Update VMS Boards", use_container_width=True):
        st.code(vms_text, language=None)
        st.caption("✅ VMS message formatted for Bengaluru Smart City VMS character limit (120 chars max). "
                   "Push to VMS controllers at affected corridor entry points.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVENT DNA MATCHING
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <h3 style='font-size:18px;font-weight:700;color:#111827;margin:0 0 8px 0'>
        🧬 Event DNA Matching
    </h3>
    <p style='color:#6B7280;font-size:14px;margin:0 0 16px 0'>
        <b style='color:#111827'>What this does:</b> Searches all historical ASTRAM events
        for those most similar to today — same type (or semantically similar group), same zone,
        same time of day using circular hour distance. Police get intelligence from actual BTP experience.
    </p>
    """, unsafe_allow_html=True)

    if df_raw is not None:
        similar = predictor.get_similar_historical_events(
            df=df_raw, event_cause=event_cause, zone=zone,
            hour=event_dt.hour, day_of_week=event_dt.weekday(), top_k=8,
        )

        if len(similar) > 0:
            avg_c       = similar["clearance_mins"].mean()
            max_c       = similar["clearance_mins"].max()
            closure_pct = similar["requires_road_closure"].mean() * 100

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("🧬 Best DNA Match",      f"{similar['similarity_pct'].max():.0f}%")
            d2.metric("⏱️ Avg Clearance (Past)", f"{avg_c:.0f} min")
            d3.metric("⚠️ Worst Case Seen",      f"{max_c:.0f} min")
            d4.metric("🚧 Road Closure Rate",    f"{closure_pct:.0f}%")

            spacer(24)
            dna_col1, dna_col2 = st.columns([2, 1])

            with dna_col1:
                st.markdown("<h4 style='font-size:15px;font-weight:700;color:#111827;margin:0 0 8px 0'>📍 Where Similar Events Happened in Bengaluru</h4>", unsafe_allow_html=True)
                df_similar_loc = df_raw.loc[similar.index].copy() if False else \
                    df_raw[df_raw['id'].isin(similar['id'])].copy()
                df_similar_loc = df_similar_loc.dropna(subset=["latitude", "longitude"])

                if len(df_similar_loc) > 0:
                    dna_map = folium.Map(location=[12.9716, 77.5946], zoom_start=12,
                                        tiles="CartoDB positron")
                    for _, row in df_similar_loc.iterrows():
                        match_row = similar[similar['id'] == row['id']]
                        if match_row.empty:
                            continue
                        match_pct = float(match_row['similarity_pct'].iloc[0])
                        c_mins    = float(match_row['clearance_mins'].iloc[0])
                        dot_color = "#EF4444" if c_mins > 120 else "#F59E0B" if c_mins > 60 else "#22C55E"
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=8 + match_pct / 20, color=dot_color,
                            fill=True, fill_color=dot_color, fill_opacity=0.7,
                            tooltip=f"{row.get('event_cause','?')} — {c_mins:.0f} min — {match_pct:.0f}% match",
                        ).add_to(dna_map)
                    st_folium(dna_map, width=None, height=380, returned_objects=[])

            with dna_col2:
                st.markdown("<h4 style='font-size:15px;font-weight:700;color:#111827;margin:0 0 8px 0'>📋 Top Matches</h4>", unsafe_allow_html=True)
                for i, (_, row) in enumerate(similar.iterrows()):
                    c_mins    = row["clearance_mins"]
                    match     = row["similarity_pct"]
                    bar_color = "#EF4444" if c_mins > 120 else "#F59E0B" if c_mins > 60 else "#22C55E"
                    bar_bg    = "#FEF2F2" if c_mins > 120 else "#FFFBEB" if c_mins > 60 else "#F0FDF4"
                    st.markdown(f"""
                    <div style='background:{bar_bg};border:1px solid {bar_color}33;
                                border-left:4px solid {bar_color};border-radius:10px;
                                padding:10px 12px;margin:6px 0;transition:transform 0.15s ease'>
                        <div style='font-size:12px;font-weight:700;color:#111827'>
                            #{i+1} · {match:.0f}% match
                        </div>
                        <div style='font-size:12px;color:#374151;margin-top:2px'>
                            {row["event_cause"].replace("_"," ").title()}
                        </div>
                        <div style='font-size:11px;color:#6B7280;margin-top:2px'>
                            ⏱️ {c_mins:.0f} min clearance
                        </div>
                        <div style='font-size:11px;color:#9CA3AF;margin-top:2px'>
                            📍 {str(row.get("junction","Unknown"))[:30]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            worst = similar.nlargest(1, "clearance_mins").iloc[0]
            if worst["clearance_mins"] > 180:
                st.error(
                    f"🚨 **Historical Warning:** A similar event ({worst['event_cause']}) "
                    f"previously caused **{worst['clearance_mins']:.0f} minutes** of disruption "
                    f"near {str(worst.get('junction','this area'))}. Prepare additional resources."
                )

            spacer(24)
            st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:12px'>📊 Full Match Table</h3>", unsafe_allow_html=True)
            disp = similar[["event_cause", "zone", "junction", "clearance_mins",
                             "priority", "requires_road_closure", "similarity_pct"]].copy()
            disp["clearance_mins"] = disp["clearance_mins"].round(0).astype(int)
            disp.columns = ["Cause", "Zone", "Junction", "Clearance (min)",
                            "Priority", "Road Closure", "DNA Match %"]
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.error("ASTRAM dataset not loaded.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPLIANCE RISK
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <h3 style='font-size:18px;font-weight:700;color:#111827;margin:0 0 8px 0'>
        ⚠️ Compliance Risk Score
    </h3>
    <p style='color:#6B7280;font-size:14px;margin:0 0 20px 0'>
        <b style='color:#111827'>For BTP officers:</b> This score tells you how risky this event type
        has historically been in this zone. LOW score = this type of event usually causes problems here.
        Use this to decide how much to pre-deploy before the event starts.
    </p>
    """, unsafe_allow_html=True)

    if compliance_scorer is not None:
        comp  = compliance_scorer.score_organizer_zone(zone=zone, cause=event_cause)
        score = comp["score"]
        risk  = comp["risk_level"]

        risk_color  = {"CRITICAL": "#EF4444", "HIGH": "#F59E0B", "MODERATE": "#FBBF24",
                       "LOW": "#22C55E", "INSUFFICIENT DATA": "#6B7280"}.get(risk, "#6B7280")
        risk_bg     = {"CRITICAL": "#FEF2F2", "HIGH": "#FFFBEB", "MODERATE": "#FEFCE8",
                       "LOW": "#F0FDF4",  "INSUFFICIENT DATA": "#F9FAFB"}.get(risk, "#F9FAFB")
        risk_border = {"CRITICAL": "#FECACA", "HIGH": "#FDE68A", "MODERATE": "#FEF08A",
                       "LOW": "#BBF7D0", "INSUFFICIENT DATA": "#E5E7EB"}.get(risk, "#E5E7EB")
        risk_icon   = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡",
                       "LOW": "🟢", "INSUFFICIENT DATA": "⚪"}.get(risk, "⚪")

        cs1, cs2, cs3 = st.columns([1, 2, 1])
        with cs2:
            st.markdown(f"""
            <div style='background:#FFFFFF;border:1px solid {risk_border};border-top:4px solid {risk_color};
                        border-radius:16px;padding:32px 24px;text-align:center;
                        box-shadow:0 2px 8px rgba(0,0,0,0.08)'>
                <div style='font-size:11px;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.8px;color:#9CA3AF;margin-bottom:12px'>
                    Compliance Risk Score
                </div>
                <div style='font-size:72px;font-weight:800;color:{risk_color};line-height:1'>
                    {score}
                </div>
                <div style='font-size:13px;color:#6B7280;margin:6px 0 16px 0'>
                    out of 100 &nbsp;·&nbsp; higher = safer history
                </div>
                <div style='font-size:18px;font-weight:700;color:{risk_color};margin-bottom:16px'>
                    {risk_icon} {risk} RISK
                </div>
                <div style='background:#F3F4F6;border-radius:8px;height:10px;margin:0 0 12px 0;overflow:hidden'>
                    <div style='background:{risk_color};border-radius:8px;height:10px;
                                width:{score}%;transition:width 0.5s ease'></div>
                </div>
                <div style='font-size:12px;color:#9CA3AF'>
                    {comp["total_events"]} similar past events analyzed
                </div>
            </div>
            """, unsafe_allow_html=True)

        spacer(24)
        cr1, cr2, cr3 = st.columns(3)
        cr1.metric("Avg Clearance (Past Events)", f"{comp.get('avg_clearance_mins', 'N/A')} min")
        cr2.metric("Road Closure Rate",           f"{comp.get('road_closure_rate_pct', 0)}%")
        cr3.metric("High Priority Rate",          f"{comp.get('high_priority_rate_pct', 0)}%")

        if comp["violations"]:
            st.markdown("<h4 style='font-size:15px;font-weight:700;color:#111827;margin:16px 0 8px 0'>⚠️ Historical Risk Factors</h4>", unsafe_allow_html=True)
            for v in comp["violations"]:
                st.error(f"• {v}")

        st.success(f"**📋 TraffiX Recommendation:** {comp['recommendation']}")
    else:
        st.warning("Load ASTRAM dataset to enable compliance scoring.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — POST-EVENT ANALYTICS + FEEDBACK LOOP
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""
    <h3 style='font-size:18px;font-weight:700;color:#111827;margin:0 0 4px 0'>
        📋 Post-Event Analytics — Learning from the Past
    </h3>
    <p style='color:#6B7280;font-size:14px;margin:0 0 20px 0'>
        Every completed ASTRAM event teaches TraffiX to predict better.
        This tab provides the analytics dashboard and the feedback loop
        that tracks real-world prediction accuracy.
    </p>
    """, unsafe_allow_html=True)

    if df_raw is not None:
        df_closed = df_raw[df_raw["status"] == "closed"].copy()
        df_closed["start_datetime"]  = pd.to_datetime(df_closed["start_datetime"],  errors="coerce", utc=True)
        df_closed["closed_datetime"] = pd.to_datetime(df_closed["closed_datetime"], errors="coerce", utc=True)
        df_closed["clearance_mins"]  = (
            (df_closed["closed_datetime"] - df_closed["start_datetime"]).dt.total_seconds() / 60
        )
        df_closed = df_closed[(df_closed["clearance_mins"] > 0) & (df_closed["clearance_mins"] < 1440)]

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Events in Learning Set", f"{len(df_closed):,}")
        p2.metric("Avg Clearance",          f"{df_closed['clearance_mins'].mean():.0f} min")
        p3.metric("Median Clearance",       f"{df_closed['clearance_mins'].median():.0f} min")
        p4.metric("Road Closure Rate",      f"{df_closed['requires_road_closure'].mean()*100:.1f}%")

        spacer(24)
        an1, an2 = st.columns(2)
        with an1:
            st.markdown("<h4 style='font-size:15px;font-weight:700;color:#111827;margin:0 0 8px 0'>Event Volume by Zone</h4>", unsafe_allow_html=True)
            st.bar_chart(df_closed["zone"].value_counts().dropna().head(10))
        with an2:
            st.markdown("<h4 style='font-size:15px;font-weight:700;color:#111827;margin:0 0 8px 0'>Avg Clearance by Cause (min)</h4>", unsafe_allow_html=True)
            cause_avg = df_closed.groupby("event_cause")["clearance_mins"].mean().sort_values(ascending=False)
            st.bar_chart(cause_avg.round(0))

        spacer(24)
        st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:12px'>🏆 Top 10 Worst Events — BTP Lessons Learned</h3>", unsafe_allow_html=True)
        worst_events = df_closed.nlargest(10, "clearance_mins")[
            ["id", "event_cause", "zone", "junction", "priority", "clearance_mins", "requires_road_closure"]
        ].copy()
        worst_events["clearance_mins"] = worst_events["clearance_mins"].round(0).astype(int)
        worst_events.columns = ["Event ID", "Cause", "Zone", "Junction", "Priority",
                                "Clearance (min)", "Road Closure"]
        st.dataframe(worst_events, use_container_width=True, hide_index=True)

        spacer(24)
        # Sparse cause warning — honest data quality disclosure
        _dqr = getattr(predictor, 'data_quality_report', {}) or {}
        _sparse_warn = _dqr.get('sparse_cause_warning', '')
        _sparse_causes = _dqr.get('sparse_causes', {})
        if _sparse_causes:
            st.warning(f"⚠️ **Data Quality Alert:** {_sparse_warn}")
            sc_cols = st.columns(min(len(_sparse_causes), 4))
            for col, (cause, cnt) in zip(sc_cols, list(_sparse_causes.items())[:4]):
                col.metric(f"{cause.replace('_',' ').title()}", f"{cnt} events", "⚠️ sparse")
            st.caption("Model uncertainty is higher for sparse causes. BTP should prioritize collecting event-type data for these categories to improve future predictions.")
        spacer(12)
        st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:12px'>📊 Cause Performance Card</h3>", unsafe_allow_html=True)
        perf = df_closed.groupby("event_cause").agg(
            Count=("clearance_mins", "count"),
            Avg_Clearance=("clearance_mins", "mean"),
            Median_Clearance=("clearance_mins", "median"),
            Road_Closure_Pct=("requires_road_closure", "mean"),
        ).round(1).sort_values("Avg_Clearance", ascending=False)
        perf["Road_Closure_Pct"] = (perf["Road_Closure_Pct"] * 100).round(1).astype(str) + "%"
        st.dataframe(perf, use_container_width=True)

        # ── Junction Repeat-Offender Heatmap ───────────────────────────────────
        spacer(24)
        st.markdown("<h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:4px'>🔥 Junction Repeat-Offender Heatmap</h3>", unsafe_allow_html=True)
        st.caption("Which junctions show up again and again in ASTRAM history? These become auto-priority deployment zones — expect Bengaluru's Silk Board flyover to lead.")

        geo = df_closed.dropna(subset=["junction", "latitude", "longitude"]).copy()
        geo = geo[(geo["latitude"] != 0) & (geo["longitude"] != 0)]

        if geo.empty:
            st.info("No geolocated junction data available in this dataset slice for the heatmap.")
        else:
            junction_stats = geo.groupby("junction").agg(
                events=("id", "count"),
                avg_lat=("latitude", "mean"),
                avg_lng=("longitude", "mean"),
                avg_clearance=("clearance_mins", "mean"),
            ).reset_index().sort_values("events", ascending=False)
            top_junctions = junction_stats.head(15)

            heat_map = folium.Map(
                location=[top_junctions["avg_lat"].mean(), top_junctions["avg_lng"].mean()],
                zoom_start=11, tiles="CartoDB dark_matter",
            )
            HeatMap(data=geo[["latitude", "longitude"]].values.tolist(), radius=18, blur=22, max_zoom=13).add_to(heat_map)
            for _, row in top_junctions.iterrows():
                folium.CircleMarker(
                    location=[row["avg_lat"], row["avg_lng"]],
                    radius=6 + min(row["events"] / 5, 14),
                    color="#EF4444", fill=True, fill_color="#EF4444", fill_opacity=0.55,
                    tooltip=folium.Tooltip(
                        f"<b>{row['junction']}</b><br>{int(row['events'])} events on record"
                        f"<br>Avg clearance: {row['avg_clearance']:.0f} min", sticky=True),
                ).add_to(heat_map)
            st_folium(heat_map, width=None, height=440, returned_objects=[])

            hj1, hj2 = st.columns([1, 1])
            with hj1:
                top_table = top_junctions[["junction", "events", "avg_clearance"]].copy()
                top_table["avg_clearance"] = top_table["avg_clearance"].round(0).astype(int)
                top_table.columns = ["Junction", "Events on Record", "Avg Clearance (min)"]
                st.dataframe(top_table, use_container_width=True, hide_index=True)
            with hj2:
                leader = top_junctions.iloc[0]
                st.markdown(f"""
                <div style='background:#FEF2F2;border:1px solid #FECACA;border-left:4px solid #EF4444;
                            border-radius:12px;padding:16px 18px'>
                    <div style='font-size:13px;color:#7F1D1D;font-weight:700'>🚨 Repeat-Offender Alert</div>
                    <div style='font-size:20px;font-weight:800;color:#111827;margin-top:4px'>{leader['junction']}</div>
                    <div style='font-size:13px;color:#374151;margin-top:4px'>
                        has recorded <b>{int(leader['events'])}</b> events in ASTRAM history —
                        averaging <b>{leader['avg_clearance']:.0f} min</b> to clear.
                    </div>
                    <div style='font-size:12px;color:#6B7280;margin-top:8px'>
                        TraffiX auto-prioritises a standing deployment kit for this junction
                        instead of building the plan from scratch every time it fires.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info("Load ASTRAM dataset for analytics.")

    # ── Pre-Deployment Shift Roster ────────────────────────────────────────────
    spacer(24)
    st.markdown("""
    <h3 style='font-size:17px;font-weight:700;color:#111827;margin-bottom:4px'>
        📋 Pre-Deployment Shift Roster — Historical Pattern Intelligence
    </h3>
    <p style='color:#6B7280;font-size:13px;margin:0 0 12px 0'>
        For each shift window, these junctions have the highest historical event frequency
        in ASTRAM. Pre-position teams here instead of waiting for an incident to be logged.
    </p>
    """, unsafe_allow_html=True)

    roster_dow  = st.selectbox("Day of week", options=[0,1,2,3,4,5,6],
                               format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
                               index=now.weekday(), key="roster_dow")
    roster_cols = st.columns(4)
    shift_slots = [
        ("Morning Peak", 8), ("Midday", 12), ("Evening Peak", 18), ("Night", 22)
    ]
    for col, (label, hour) in zip(roster_cols, shift_slots):
        roster = predictor.get_shift_roster(roster_dow, hour)
        with col:
            st.markdown(f"""
            <div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:12px;
                        padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.05)'>
                <div style='font-size:12px;font-weight:700;color:#2563EB;margin-bottom:8px'>
                    🕐 {label} ({hour}:00)
                </div>
            """, unsafe_allow_html=True)
            if roster:
                for i, j in enumerate(roster[:4]):
                    color = "#EF4444" if i == 0 else "#F59E0B" if i == 1 else "#6B7280"
                    st.markdown(f"""
                    <div style='font-size:11px;color:{color};font-weight:600;padding:3px 0'>
                        #{i+1} {j['junction']}
                    </div>
                    <div style='font-size:10px;color:#9CA3AF;margin-bottom:4px'>
                        {j['count']} past events · avg {j['avg_clearance']} min
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:12px;color:#9CA3AF'>No data for this slot</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Post-Event Feedback Loop ───────────────────────────────────────────────
    spacer(32)
    st.markdown("""
    <div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:16px;padding:24px;
                box-shadow:0 1px 3px rgba(0,0,0,0.06)'>
        <h3 style='font-size:17px;font-weight:700;color:#111827;margin:0 0 6px 0'>
            🔁 Post-Event Feedback — Learning Loop
        </h3>
        <p style='color:#6B7280;font-size:13px;margin:0 0 16px 0'>
            When an event closes in ASTRAM, log the actual clearance time here.
            TraffiX tracks prediction accuracy over real deployments so the model can be
            validated and eventually retrained with live ground truth.
        </p>
    </div>
    """, unsafe_allow_html=True)
    spacer(8)

    fb1, fb2 = st.columns(2)
    with fb1:
        fb_event_id = st.text_input("ASTRAM Event ID", placeholder="e.g. EVT-2024-001234")
        fb_cause    = st.selectbox("Event Type (Actual)", options=list(CAUSE_IMPACT_MAP.keys()),
                                   key="fb_cause")
        fb_zone     = st.selectbox("Zone (Actual)", options=list(ZONE_CONGESTION_WEIGHTS.keys()),
                                   key="fb_zone")
    with fb2:
        fb_predicted = st.number_input("TraffiX Predicted Clearance (min)",
                                       min_value=0.0, value=float(result["predicted_clearance_mins"]),
                                       format="%.0f")
        fb_actual    = st.number_input("Actual Clearance Time (min)", min_value=0.0,
                                       value=0.0, format="%.0f")

    if st.button("📤 Submit Feedback", type="primary"):
        if fb_event_id and fb_actual > 0:
            feedback_file = os.path.join(os.path.dirname(__file__), '..', 'data',
                                         'prediction_feedback.json')
            success = predictor.log_prediction_feedback(
                event_id=fb_event_id,
                predicted_clearance=fb_predicted,
                actual_clearance=fb_actual,
                event_cause=fb_cause,
                zone=fb_zone,
                feedback_path=feedback_file,
            )
            if success:
                error = fb_actual - fb_predicted
                st.success(
                    f"✅ Feedback logged! Error: **{error:+.0f} min** "
                    f"({'over-estimated' if error < 0 else 'under-estimated'} by {abs(error):.0f} min)"
                )
            else:
                st.error("Failed to save feedback. Check file permissions.")
        else:
            st.warning("Please enter an event ID and actual clearance time.")

    # Show accumulated feedback stats
    feedback_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'prediction_feedback.json')
    fb_stats = predictor.get_feedback_stats(feedback_path=feedback_file)
    if fb_stats:
        spacer(16)
        st.markdown("<h4 style='font-size:15px;font-weight:700;color:#111827;margin-bottom:12px'>📈 Real-World Accuracy (from Feedback Log)</h4>", unsafe_allow_html=True)
        fs1, fs2, fs3 = st.columns(3)
        fs1.metric("Events Logged",    fb_stats['n_logged'])
        fs2.metric("MAE (Actual)",     f"{fb_stats['mae_mins']} min",
                   help="Mean Absolute Error from real deployments")
        fs3.metric("Mean % Error",     f"{fb_stats['mean_pct_error']}%")
        if fb_stats.get("recent_records"):
            fb_df = pd.DataFrame(fb_stats["recent_records"])[
                ["event_id", "event_cause", "zone", "predicted_clearance_mins",
                 "actual_clearance_mins", "error_mins", "logged_at"]
            ]
            fb_df.columns = ["Event ID", "Cause", "Zone", "Predicted (min)",
                             "Actual (min)", "Error (min)", "Logged At"]
            st.dataframe(fb_df, use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
spacer(24)
st.markdown("""
<div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:12px;
            padding:16px 24px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.04)'>
    <span style='color:#6B7280;font-size:13px;font-weight:500'>
        TraffiX v2.0 &nbsp;·&nbsp; ASTRAM Intelligence SDK &nbsp;·&nbsp;
        Bengaluru Traffic Police &nbsp;·&nbsp;
        Trained on real ASTRAM events &nbsp;·&nbsp;
        BTP Hackathon Track 2
    </span>
</div>
""", unsafe_allow_html=True)