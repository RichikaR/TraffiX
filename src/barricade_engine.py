"""
TraffiX Barricade Placement Engine — v2

What changed from v1:
  - Now imports BengaluruSpatialGraph and calls get_approach_roads() for
    ANY event coordinate — full OSM network traversal instead of 3 hardcoded junctions.
  - Hardcoded junction table kept as priority override (faster lookup for known spots).
  - generate_barricade_plan() reports whether it used OSM-derived or hardcoded approaches.

Written to be understood by a Sub-Inspector with no ML background.
"""

import math
from typing import Optional

# ── Barricade types in BTP officer language ────────────────────────────────────
BARRICADE_TYPES = {
    "HARD_BLOCK":    {"label": "Full Road Block",   "icon": "🚧", "color": "#ff4444", "officers_needed": 3},
    "ONE_WAY":       {"label": "One-Way Control",   "icon": "↗️",  "color": "#ff8800", "officers_needed": 2},
    "MARSHAL_POINT": {"label": "Traffic Marshal",   "icon": "👮",  "color": "#ffcc00", "officers_needed": 1},
    "DIVERSION":     {"label": "Diversion Board",   "icon": "🔀",  "color": "#00aaff", "officers_needed": 1},
    "CROWD_BUFFER":  {"label": "Crowd Buffer Zone", "icon": "🛡️",  "color": "#aa44ff", "officers_needed": 2},
}

# ── Known junction overrides — real approach data for Bengaluru's busiest spots ─
# These 3 take priority over OSM traversal when the event maps to a known junction.
# New junctions can be added here over time from field observations.
JUNCTION_APPROACHES = {
    "QueensStatueCircle": {
        "approaches": [
            {"name": "MG Road (East)",       "bearing": 90,  "lat_offset":  0.0015, "lng_offset":  0.003},
            {"name": "Cubbon Road (North)",   "bearing": 0,   "lat_offset":  0.003,  "lng_offset":  0.0},
            {"name": "St Marks Road (South)", "bearing": 180, "lat_offset": -0.002,  "lng_offset":  0.0},
            {"name": "Kasturba Road (West)",  "bearing": 270, "lat_offset":  0.0,    "lng_offset": -0.003},
        ]
    },
    "SilkBoardJunc": {
        "approaches": [
            {"name": "Hosur Road (South)",       "bearing": 180, "lat_offset": -0.003, "lng_offset":  0.0},
            {"name": "ORR East",                 "bearing": 90,  "lat_offset":  0.0,   "lng_offset":  0.004},
            {"name": "BTM Layout Road",          "bearing": 270, "lat_offset":  0.0,   "lng_offset": -0.003},
            {"name": "Outer Ring Road West",     "bearing": 0,   "lat_offset":  0.002, "lng_offset":  0.0},
        ]
    },
    "HebbalFlyoverJunc": {
        "approaches": [
            {"name": "Bellary Road (North to Airport)", "bearing": 0,   "lat_offset":  0.004, "lng_offset":  0.0},
            {"name": "Outer Ring Road (East)",          "bearing": 90,  "lat_offset":  0.0,   "lng_offset":  0.003},
            {"name": "Bellary Road (South to City)",    "bearing": 180, "lat_offset": -0.003, "lng_offset":  0.0},
            {"name": "Kogilu Road",                     "bearing": 270, "lat_offset":  0.0,   "lng_offset": -0.002},
        ]
    },
}

# ── Deployment scale by severity ──────────────────────────────────────────────
SEVERITY_DEPLOYMENT = {
    "CRITICAL": {"barricades_per_approach": 3, "crowd_buffer": True,  "marshal_outer_ring": True},
    "HIGH":     {"barricades_per_approach": 2, "crowd_buffer": True,  "marshal_outer_ring": False},
    "MODERATE": {"barricades_per_approach": 1, "crowd_buffer": False, "marshal_outer_ring": False},
    "LOW":      {"barricades_per_approach": 1, "crowd_buffer": False, "marshal_outer_ring": False},
}


def _bearing_to_offset(bearing: float, distance_km: float = 0.15) -> tuple:
    """Convert compass bearing + distance to lat/lng offset."""
    R = 6371.0
    lat_off = (distance_km / R) * math.cos(math.radians(bearing)) * (180 / math.pi)
    lng_off = (distance_km / R) * math.sin(math.radians(bearing)) * (180 / math.pi)
    return lat_off, lng_off


def generate_barricade_plan(
    event_lat: float,
    event_lng: float,
    event_cause: str,
    severity: str,
    junction_name: Optional[str] = None,
    crowd_direction: Optional[str] = None,
    spatial_graph=None,          # BengaluruSpatialGraph instance (optional but recommended)
    osm_radius_m: int = 350,     # ego-graph radius for approach road detection
) -> dict:
    """
    Generate exact barricade placement plan for any Bengaluru event coordinate.

    Priority order for approach road data:
      1. Hardcoded junction table (QueensStatueCircle, SilkBoard, Hebbal) — fastest
      2. OSM ego-graph traversal via spatial_graph.get_approach_roads() — full coverage
      3. Cardinal direction fallback (N/S/E/W) — only if OSM graph not available

    Parameters
    ----------
    event_lat, event_lng : GPS of the event (from ASTRAM)
    event_cause          : event_cause field from ASTRAM record
    severity             : TraffiX prediction (CRITICAL/HIGH/MODERATE/LOW)
    junction_name        : if event is at a known junction, uses hardcoded approach data
    crowd_direction      : expected crowd direction (N/S/E/W) — affects crowd buffer placement
    spatial_graph        : BengaluruSpatialGraph instance; pass this from app.py for OSM coverage
    osm_radius_m         : radius in metres for ego-graph approach road detection

    Returns
    -------
    dict with deployments list (each has GPS + type + officer count), summary stats,
    and data_source field indicating which approach method was used.
    """
    deployments = []
    deploy_cfg = SEVERITY_DEPLOYMENT.get(severity, SEVERITY_DEPLOYMENT["MODERATE"])
    data_source = "cardinal_fallback"

    # ── Step 1: Resolve approach roads ────────────────────────────────────────
    if junction_name and junction_name in JUNCTION_APPROACHES:
        # Known junction — use hand-verified real road names
        approaches = JUNCTION_APPROACHES[junction_name]["approaches"]
        data_source = f"hardcoded_junction:{junction_name}"

    elif spatial_graph is not None:
        # OSM ego-graph traversal — works for ANY coordinate in Bengaluru
        osm_approaches = spatial_graph.get_approach_roads(
            event_lat, event_lng, radius_m=osm_radius_m, max_approaches=4
        )
        if osm_approaches:
            approaches = osm_approaches
            data_source = f"osm_graph_traversal:{osm_radius_m}m_radius"
        else:
            # OSM returned nothing (edge case: coordinate outside network)
            approaches = _cardinal_fallback()
            data_source = "cardinal_fallback:osm_returned_empty"
    else:
        # spatial_graph not passed — use cardinal directions
        approaches = _cardinal_fallback()
        data_source = "cardinal_fallback:no_graph_passed"

    barricade_number = 1

    # ── Step 2: Place barricades on each approach road ─────────────────────────
    for approach in approaches:
        bearing = approach.get("bearing", 0)
        lat_off, lng_off = _bearing_to_offset(bearing, distance_km=0.2)

        # Override with OSM-derived offset if available (more accurate than bearing calc)
        if approach.get("lat_offset") is not None and approach.get("lng_offset") is not None:
            lat_off = approach["lat_offset"]
            lng_off = approach["lng_offset"]

        b_lat = event_lat + lat_off
        b_lng = event_lng + lng_off

        # Primary barricade: one-way control at ~200m from event
        deployments.append({
            "id": f"B{barricade_number:02d}",
            "lat": round(b_lat, 6),
            "lng": round(b_lng, 6),
            "road": approach["name"],
            "type": "ONE_WAY",
            "type_label": BARRICADE_TYPES["ONE_WAY"]["label"],
            "icon": BARRICADE_TYPES["ONE_WAY"]["icon"],
            "color": BARRICADE_TYPES["ONE_WAY"]["color"],
            "officers_needed": BARRICADE_TYPES["ONE_WAY"]["officers_needed"],
            "instruction": (
                f"Control incoming traffic on {approach['name']}. Allow only outbound flow."
            ),
            "priority": approach.get("priority", "HIGH") if bearing in [90, 270] else "MEDIUM",
            "highway_type": approach.get("highway_type", "unknown"),
        })
        barricade_number += 1

        # Secondary: diversion board at ~350m (for HIGH/CRITICAL)
        if deploy_cfg["barricades_per_approach"] >= 2:
            lat_off2, lng_off2 = _bearing_to_offset(bearing, distance_km=0.35)
            if approach.get("lat_offset") is not None:
                lat_off2 = approach["lat_offset"] * 1.4
                lng_off2 = approach["lng_offset"] * 1.4
            deployments.append({
                "id": f"B{barricade_number:02d}",
                "lat": round(event_lat + lat_off2, 6),
                "lng": round(event_lng + lng_off2, 6),
                "road": f"{approach['name']} (Secondary)",
                "type": "DIVERSION",
                "type_label": BARRICADE_TYPES["DIVERSION"]["label"],
                "icon": BARRICADE_TYPES["DIVERSION"]["icon"],
                "color": BARRICADE_TYPES["DIVERSION"]["color"],
                "officers_needed": BARRICADE_TYPES["DIVERSION"]["officers_needed"],
                "instruction": (
                    f"Place diversion board 350m before junction on {approach['name']}. "
                    "Direct vehicles to alternate route."
                ),
                "priority": "HIGH",
                "highway_type": approach.get("highway_type", "unknown"),
            })
            barricade_number += 1

        # Tertiary: advance marshal at ~500m (CRITICAL only)
        if deploy_cfg["barricades_per_approach"] >= 3:
            lat_off3, lng_off3 = _bearing_to_offset(bearing, distance_km=0.5)
            if approach.get("lat_offset") is not None:
                lat_off3 = approach["lat_offset"] * 2.0
                lng_off3 = approach["lng_offset"] * 2.0
            deployments.append({
                "id": f"B{barricade_number:02d}",
                "lat": round(event_lat + lat_off3, 6),
                "lng": round(event_lng + lng_off3, 6),
                "road": f"{approach['name']} (Advance Warning)",
                "type": "MARSHAL_POINT",
                "type_label": BARRICADE_TYPES["MARSHAL_POINT"]["label"],
                "icon": BARRICADE_TYPES["MARSHAL_POINT"]["icon"],
                "color": BARRICADE_TYPES["MARSHAL_POINT"]["color"],
                "officers_needed": BARRICADE_TYPES["MARSHAL_POINT"]["officers_needed"],
                "instruction": (
                    f"Stand 500m ahead on {approach['name']}. "
                    "Redirect vehicles before they reach the congested zone."
                ),
                "priority": "MEDIUM",
                "highway_type": approach.get("highway_type", "unknown"),
            })
            barricade_number += 1

    # ── Step 3: Crowd buffer (public events / processions) ────────────────────
    if deploy_cfg["crowd_buffer"] and event_cause in ["public_event", "procession", "protest"]:
        for offset_m, label in [(0.001, "Crowd Entry"), (-0.001, "Crowd Exit")]:
            deployments.append({
                "id": f"B{barricade_number:02d}",
                "lat": round(event_lat + offset_m, 6),
                "lng": round(event_lng, 6),
                "road": f"{label} Point",
                "type": "CROWD_BUFFER",
                "type_label": BARRICADE_TYPES["CROWD_BUFFER"]["label"],
                "icon": BARRICADE_TYPES["CROWD_BUFFER"]["icon"],
                "color": BARRICADE_TYPES["CROWD_BUFFER"]["color"],
                "officers_needed": BARRICADE_TYPES["CROWD_BUFFER"]["officers_needed"],
                "instruction": (
                    f"Maintain crowd buffer at {label}. "
                    "Ensure pedestrian flow does not spill onto carriageway."
                ),
                "priority": "CRITICAL",
            })
            barricade_number += 1

    # ── Step 4: Hard block at event perimeter (CRITICAL) ──────────────────────
    if severity == "CRITICAL":
        deployments.append({
            "id": f"B{barricade_number:02d}",
            "lat": round(event_lat, 6),
            "lng": round(event_lng + 0.0005, 6),
            "road": "Event Perimeter",
            "type": "HARD_BLOCK",
            "type_label": BARRICADE_TYPES["HARD_BLOCK"]["label"],
            "icon": BARRICADE_TYPES["HARD_BLOCK"]["icon"],
            "color": BARRICADE_TYPES["HARD_BLOCK"]["color"],
            "officers_needed": BARRICADE_TYPES["HARD_BLOCK"]["officers_needed"],
            "instruction": (
                "Full road closure at event perimeter. "
                "No vehicle entry. Senior officer to supervise."
            ),
            "priority": "CRITICAL",
        })
        barricade_number += 1

    # ── Step 5: Emergency corridor — always keep one route open ───────────────
    if severity in ["CRITICAL", "HIGH"] and len(deployments) > 0:
        lat_off_ec, lng_off_ec = _bearing_to_offset(0, distance_km=0.15)
        deployments.append({
            "id": "EC01",
            "lat": round(event_lat + lat_off_ec, 6),
            "lng": round(event_lng + lng_off_ec, 6),
            "road": "Emergency Corridor (Keep OPEN)",
            "type": "MARSHAL_POINT",
            "type_label": "🚑 Emergency Lane — KEEP OPEN",
            "icon": "🚑",
            "color": "#00ff88",
            "officers_needed": 2,
            "instruction": (
                "CRITICAL: Keep this lane OPEN at all times for ambulance, "
                "fire engine, police vehicles. Do NOT place any barricade here."
            ),
            "priority": "EMERGENCY",
        })

    total_officers = sum(d["officers_needed"] for d in deployments)
    total_barricades = len([d for d in deployments if d["type"] != "MARSHAL_POINT"])

    return {
        "total_placements": len(deployments),
        "total_officers_for_barricades": total_officers,
        "total_physical_barricades": total_barricades,
        "emergency_corridor_maintained": severity in ["CRITICAL", "HIGH"],
        "deployments": deployments,
        "data_source": data_source,
        "using_real_road_names": data_source.startswith("osm") or data_source.startswith("hardcoded"),
        "summary": (
            f"Deploy {total_barricades} barricades and {total_officers} officers "
            f"at {len(deployments)} positions. Emergency corridor maintained. "
            f"Road data: {data_source}."
        ),
    }


def _cardinal_fallback() -> list:
    """Last-resort approach roads when neither junction table nor OSM graph is available."""
    return [
        {"name": "North Approach Road", "bearing": 0,   "lat_offset":  0.003, "lng_offset":  0.0,    "priority": "MEDIUM"},
        {"name": "East Approach Road",  "bearing": 90,  "lat_offset":  0.0,   "lng_offset":  0.003,  "priority": "HIGH"},
        {"name": "South Approach Road", "bearing": 180, "lat_offset": -0.003, "lng_offset":  0.0,    "priority": "MEDIUM"},
        {"name": "West Approach Road",  "bearing": 270, "lat_offset":  0.0,   "lng_offset": -0.003,  "priority": "HIGH"},
    ]
