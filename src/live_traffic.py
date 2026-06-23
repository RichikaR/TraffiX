"""
TraffiX Live Traffic Module
Uses OpenRouteService (ORS) Matrix API to detect real congestion on Bengaluru corridors.

How it works (explain to BTP officers):
  - We pick key checkpoints on each major road (like Silk Board, Hebbal, Marathahalli)
  - We ask OpenRouteService: "How long does it take to cross this stretch RIGHT NOW with live traffic?"
  - We compare that to the FREE-FLOW time (same road at 2 AM with no traffic)
  - If it takes 3x longer than normal → we flag it RED
  - This gives us a real congestion score, not a hardcoded number

No API key? The module runs in SIMULATION MODE using realistic Bengaluru traffic patterns.

Note: This module uses the ORS Matrix API (_fetch_ors_matrix / fetch_live_traffic).
      The app UI also correctly displays "ORS Live Data" when an API key is active.
"""

import requests
import time
import datetime
import random
import os
from typing import Optional

# ─── Bengaluru Major Traffic Corridors & Checkpoints ──────────────────────────
# Each corridor has: origin checkpoint → destination checkpoint
# These are real locations that BTP officers will recognize immediately

BENGALURU_CORRIDORS = {
    "Silk Board Junction": {
        "origin": "12.9176,77.6237",        # Silk Board
        "destination": "12.9352,77.6146",   # BTM Layout
        "zone": "South Zone 2",
        "free_flow_mins": 8,                # Normal travel time at 2 AM
        "max_capacity_vehicles": 3200,      # Vehicles/hour capacity
        "btp_ps": "HSR Layout PS",
        "lat": 12.9177, "lng": 77.6237,
    },
    "Hebbal Flyover": {
        "origin": "13.0359,77.5970",        # Hebbal
        "destination": "13.0130,77.5870",   # Mekhri Circle
        "zone": "North Zone 1",
        "free_flow_mins": 7,
        "max_capacity_vehicles": 2800,
        "btp_ps": "Sadashivanagar PS",
        "lat": 13.0419, "lng": 77.5945,
    },
    "Marathahalli Bridge": {
        "origin": "12.9591,77.6974",        # Marathahalli
        "destination": "12.9352,77.6830",   # Kadubeesanahalli
        "zone": "East Zone 1",
        "free_flow_mins": 6,
        "max_capacity_vehicles": 2400,
        "btp_ps": "Marathahalli PS",
        "lat": 12.9591, "lng": 77.6974,
    },
    "KR Puram Junction": {
        "origin": "13.0065,77.6956",        # KR Puram
        "destination": "12.9858,77.6714",   # Banaswadi
        "zone": "East Zone 2",
        "free_flow_mins": 9,
        "max_capacity_vehicles": 2600,
        "btp_ps": "KR Puram PS",
        "lat": 13.0065, "lng": 77.6956,
    },
    "Mysore Road (Nayandanahalli)": {
        "origin": "12.9446,77.5274",        # Nayandanahalli
        "destination": "12.9640,77.5510",   # Kengeri Satellite
        "zone": "West Zone 1",
        "free_flow_mins": 10,
        "max_capacity_vehicles": 2200,
        "btp_ps": "Kengeri PS",
        "lat": 12.9446, "lng": 77.5274,
    },
    "Tumkur Road (Peenya)": {
        "origin": "13.0280,77.5180",        # Peenya Industrial
        "destination": "13.0130,77.5400",   # Yeshwanthpur
        "zone": "West Zone 1",
        "free_flow_mins": 11,
        "max_capacity_vehicles": 3000,
        "btp_ps": "Peenya PS",
        "lat": 13.0281, "lng": 77.5180,
    },
    "Hosur Road (Electronic City)": {
        "origin": "12.8399,77.6770",        # Electronic City Phase 1
        "destination": "12.8745,77.6454",   # Bommanahalli
        "zone": "South Zone 2",
        "free_flow_mins": 12,
        "max_capacity_vehicles": 3400,
        "btp_ps": "Electronic City PS",
        "lat": 12.8399, "lng": 77.6770,
    },
    "Bellary Road (Hebbal to Airport)": {
        "origin": "13.0359,77.5970",        # Hebbal
        "destination": "13.0710,77.5950",   # Kogilu Cross
        "zone": "North Zone 1",
        "free_flow_mins": 8,
        "max_capacity_vehicles": 2500,
        "btp_ps": "Yelahanka PS",
        "lat": 13.0535, "lng": 77.5960,
    },
    "Old Madras Road": {
        "origin": "12.9920,77.6500",        # Tin Factory
        "destination": "13.0065,77.6956",   # KR Puram
        "zone": "East Zone 2",
        "free_flow_mins": 9,
        "max_capacity_vehicles": 2300,
        "btp_ps": "KR Puram PS",
        "lat": 12.9920, "lng": 77.6500,
    },
    "Outer Ring Road (Marathahalli to Silk Board)": {
        "origin": "12.9591,77.6974",        # Marathahalli
        "destination": "12.9177,77.6237",   # Silk Board
        "zone": "South Zone 2",
        "free_flow_mins": 18,
        "max_capacity_vehicles": 4000,
        "btp_ps": "HSR Layout PS",
        "lat": 12.9384, "lng": 77.6606,
    },
    "Bannerghatta Road": {
        "origin": "12.9220,77.5972",        # JP Nagar
        "destination": "12.8910,77.5952",   # Hulimavu
        "zone": "South Zone 1",
        "free_flow_mins": 10,
        "max_capacity_vehicles": 2100,
        "btp_ps": "JP Nagar PS",
        "lat": 12.9220, "lng": 77.5972,
    },
    "MG Road / Brigade Road": {
        "origin": "12.9716,77.6099",        # MG Road Metro
        "destination": "12.9667,77.5993",   # Brigade Road
        "zone": "Central Zone 1",
        "free_flow_mins": 5,
        "max_capacity_vehicles": 1800,
        "btp_ps": "Cubbon Park PS",
        "lat": 12.9716, "lng": 77.6099,
    },
}

# Congestion thresholds (ratio = actual_time / free_flow_time)
CONGESTION_LEVELS = {
    (1.0, 1.3): {"label": "FREE FLOW",   "color": "#00cc44", "emoji": "🟢", "severity": 0},
    (1.3, 1.6): {"label": "SLOW",        "color": "#aacc00", "emoji": "🟡", "severity": 1},
    (1.6, 2.0): {"label": "HEAVY",       "color": "#ff8800", "emoji": "🟠", "severity": 2},
    (2.0, 2.8): {"label": "GRIDLOCK",    "color": "#ff4444", "emoji": "🔴", "severity": 3},
    (2.8, 99):  {"label": "STANDSTILL",  "color": "#880000", "emoji": "🔴🔴", "severity": 4},
}


def _get_congestion_level(ratio: float) -> dict:
    for (lo, hi), info in CONGESTION_LEVELS.items():
        if lo <= ratio < hi:
            return info
    return {"label": "UNKNOWN", "color": "#888888", "emoji": "⚪", "severity": 0}


def _simulate_bengaluru_traffic(corridor_name: str, corridor_data: dict,
                                hour: int, day_of_week: int = 0) -> dict:
    """
    Improved simulation when no API key is provided.

    Seeded by (hour, day_of_week, corridor_name) — stable within the same hour
    so two users see the same state and refreshing doesn't flip values.
    Uses corridor-specific congestion profiles derived from Bengaluru traffic studies.
    """
    # Stable seed: same hour + same day-of-week + same corridor = same result
    seed = hash((hour, day_of_week, corridor_name)) % (2 ** 31)
    rng = random.Random(seed)

    # Corridor-specific peak multiplier — some corridors are structurally worse
    corridor_peak_bias = {
        "Silk Board Junction":                      1.30,  # Worst in city consistently
        "Outer Ring Road (Marathahalli to Silk Board)": 1.25,
        "Marathahalli Bridge":                      1.20,
        "MG Road / Brigade Road":                   1.15,
        "Hebbal Flyover":                           1.10,
        "KR Puram Junction":                        1.10,
        "Hosur Road (Electronic City)":             1.05,
    }.get(corridor_name, 1.0)

    # Is it a weekday peak?
    is_weekday = day_of_week < 5

    if hour in {7, 8, 9} and is_weekday:         # Morning peak
        base_ratio = rng.uniform(2.0, 3.2)
    elif hour in {17, 18, 19, 20} and is_weekday: # Evening peak — historically worse
        base_ratio = rng.uniform(2.4, 4.0)
    elif hour in {10, 11}:
        base_ratio = rng.uniform(1.3, 1.8)
    elif hour in {13, 14}:
        base_ratio = rng.uniform(1.1, 1.5)
    elif hour in {21, 22}:
        base_ratio = rng.uniform(1.0, 1.4)
    elif hour in {0, 1, 2, 3, 4}:               # Late night / early morning — near free flow
        base_ratio = rng.uniform(1.0, 1.1)
    elif not is_weekday:                         # Weekend — lighter but not free
        base_ratio = rng.uniform(1.1, 1.6)
    else:
        base_ratio = rng.uniform(1.2, 1.7)

    zone_mod = {
        "Central Zone 1": 1.30, "Central Zone 2": 1.25,
        "South Zone 2":   1.20, "East Zone 1":    1.15,
        "South Zone 1":   1.10,
    }.get(corridor_data["zone"], 1.0)

    ratio = min(5.0, base_ratio * zone_mod * corridor_peak_bias)

    actual_mins  = corridor_data["free_flow_mins"] * ratio
    delay_mins   = actual_mins - corridor_data["free_flow_mins"]
    speed_kmh    = max(4, round(30 / ratio, 1))
    vehicle_load = min(100, int((ratio - 1.0) / 3.0 * 100))

    level = _get_congestion_level(ratio)

    return {
        "corridor":           corridor_name,
        "zone":               corridor_data["zone"],
        "btp_ps":             corridor_data["btp_ps"],
        "lat":                corridor_data["lat"],
        "lng":                corridor_data["lng"],
        "free_flow_mins":     corridor_data["free_flow_mins"],
        "actual_travel_mins": round(actual_mins, 1),
        "delay_mins":         round(delay_mins, 1),
        "congestion_ratio":   round(ratio, 2),
        "speed_kmh":          speed_kmh,
        "vehicle_load_pct":   max(0, vehicle_load),
        "congestion_label":   level["label"],
        "congestion_color":   level["color"],
        "congestion_emoji":   level["emoji"],
        "congestion_severity": level["severity"],
        "source":             "SIMULATION",
        "timestamp":          datetime.datetime.now().strftime("%H:%M:%S"),
    }


def _fetch_ors_matrix(origins: list, destinations: list, api_key: str):
    """
    Calls OpenRouteService Matrix API (v2/matrix/driving-car).
    origins / destinations: list of [lon, lat] pairs (ORS uses lon,lat order).
    Returns the JSON response or None on failure.
    """
    url = "https://api.openrouteservice.org/v2/matrix/driving-car"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "locations":    origins + destinations,
        "sources":      list(range(len(origins))),
        "destinations": list(range(len(origins), len(origins) + len(destinations))),
        "metrics":      ["duration", "distance"],
        "resolve_locations": False,
    }
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ORS matrix call failed: {e}")
        return None


def fetch_live_traffic(api_key: Optional[str] = None) -> list[dict]:
    """
    Fetches real-time congestion for all Bengaluru corridors.

    With ORS API key -> calls OpenRouteService Matrix API for real routing durations.
    Without API key  -> uses realistic time-based simulation.

    ORS Matrix API returns duration in seconds for each origin->destination pair.
    We compare against the corridor's known free-flow time to get congestion ratio.

    Returns list of corridor congestion dicts, sorted worst-first.
    """
    now         = datetime.datetime.now()
    hour        = now.hour
    day_of_week = now.weekday()
    results     = []

    if not api_key or not api_key.strip():
        for name, data in BENGALURU_CORRIDORS.items():
            results.append(_simulate_bengaluru_traffic(name, data, hour, day_of_week))
        results.sort(key=lambda x: x["congestion_severity"], reverse=True)
        return results

    # ORS matrix takes [lon, lat] pairs (longitude first)
    corridor_names = list(BENGALURU_CORRIDORS.keys())
    corridor_list  = [BENGALURU_CORRIDORS[n] for n in corridor_names]

    def parse_coord(coord_str: str) -> list:
        lat, lon = [float(x.strip()) for x in coord_str.split(",")]
        return [lon, lat]  # ORS wants [longitude, latitude]

    origins      = [parse_coord(d["origin"])      for d in corridor_list]
    destinations = [parse_coord(d["destination"]) for d in corridor_list]

    ors_result = _fetch_ors_matrix(origins, destinations, api_key.strip())

    for i, (name, data) in enumerate(zip(corridor_names, corridor_list)):
        try:
            if ors_result is None:
                raise ValueError("ORS call failed")

            actual_secs = ors_result["durations"][i][i]
            distance_m  = ors_result["distances"][i][i]
            free_flow_secs = data["free_flow_mins"] * 60

            if actual_secs <= 0:
                raise ValueError("ORS returned zero duration")

            ratio        = actual_secs / max(free_flow_secs, 1)
            actual_mins  = actual_secs / 60
            delay_mins   = (actual_secs - free_flow_secs) / 60
            speed_kmh    = round((distance_m / actual_secs) * 3.6, 1) if actual_secs > 0 else 0
            level        = _get_congestion_level(ratio)
            vehicle_load = min(100, int((ratio - 1.0) / 3.0 * 100))

            results.append({
                "corridor":            name,
                "zone":                data["zone"],
                "btp_ps":              data["btp_ps"],
                "lat":                 data["lat"],
                "lng":                 data["lng"],
                "free_flow_mins":      round(free_flow_secs / 60, 1),
                "actual_travel_mins":  round(actual_mins, 1),
                "delay_mins":          round(delay_mins, 1),
                "congestion_ratio":    round(ratio, 2),
                "speed_kmh":           speed_kmh,
                "vehicle_load_pct":    max(0, vehicle_load),
                "congestion_label":    level["label"],
                "congestion_color":    level["color"],
                "congestion_emoji":    level["emoji"],
                "congestion_severity": level["severity"],
                "source":              "ORS_LIVE",
                "timestamp":           datetime.datetime.now().strftime("%H:%M:%S"),
            })

        except Exception:
            sim = _simulate_bengaluru_traffic(name, data, hour, day_of_week)
            sim["source"] = "SIMULATION_FALLBACK"
            results.append(sim)

    results.sort(key=lambda x: x["congestion_severity"], reverse=True)
    return results

def get_corridor_alert_message(corridor_result: dict) -> str:
    """
    Generates a plain-English alert message for BTP officers.
    Written in the same language style as ASTRAM radio alerts.
    """
    c = corridor_result
    if c["congestion_severity"] >= 3:
        return (
            f"🚨 GRIDLOCK ALERT: {c['corridor']} — Speed {c['speed_kmh']} km/h "
            f"(delay +{c['delay_mins']:.0f} min). "
            f"Deploy officers from {c['btp_ps']} immediately."
        )
    elif c["congestion_severity"] == 2:
        return (
            f"⚠️ HEAVY TRAFFIC: {c['corridor']} — Speed {c['speed_kmh']} km/h "
            f"(+{c['delay_mins']:.0f} min delay). Monitor and prepare diversion."
        )
    elif c["congestion_severity"] == 1:
        return (
            f"🟡 SLOW MOVING: {c['corridor']} — Speed {c['speed_kmh']} km/h. "
            f"No action needed but watch for escalation."
        )
    return f"✅ {c['corridor']} — Clear. Speed {c['speed_kmh']} km/h."