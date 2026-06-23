"""
TraffiX Field Dispatch Card
──────────────────────────────
Generates the one-page "field order" a Sub-Inspector would actually hand to
constables: dispatch ID, event/time/location, top resources to deploy,
barricade positions in plain language, and an emergency-corridor note.

Two outputs from the same inputs:
  generate_dispatch_html(...)  → clean HTML card for in-app preview
  generate_dispatch_pdf(...)   → print-ready one-page A4 PDF (bytes), via fpdf2

Both are deliberately built from plain dicts/lists (no Streamlit imports here)
so this module stays easy to unit-test and reuse outside the dashboard.
"""

from __future__ import annotations
import datetime
from typing import Optional

from fpdf import FPDF

SEVERITY_COLORS = {
    "CRITICAL": (239, 68, 68),
    "HIGH": (245, 158, 11),
    "MODERATE": (234, 179, 8),
    "LOW": (34, 197, 94),
}
SEVERITY_COLORS_HEX = {
    "CRITICAL": "#EF4444", "HIGH": "#F59E0B",
    "MODERATE": "#EAB308", "LOW": "#22C55E",
}

# fpdf2's core Helvetica font only supports latin-1. Free-text fields (road
# names, instructions, etc.) sometimes carry typographic Unicode punctuation
# (em-dash, smart quotes …) that would otherwise raise FPDFUnicodeEncodingException.
_PDF_UNICODE_REPLACEMENTS = {
    "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
    "\u2192": "->", "\u2265": ">=", "\u2264": "<=", "\u2022": "-",
}


def _pdf_safe(text) -> str:
    text = str(text)
    for bad, good in _PDF_UNICODE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    # Final safety net — silently drop anything still outside latin-1 rather
    # than crash the whole dispatch card over one stray character.
    return text.encode("latin-1", "replace").decode("latin-1")


# ─── Shared content builders (used by both HTML and PDF outputs) ──────────────
def make_dispatch_id(event_dt: datetime.datetime, zone: str) -> str:
    zone_code = "".join(w[0] for w in zone.split() if w).upper()[:4] or "ZN"
    return f"AEGIS-{event_dt.strftime('%Y%m%d-%H%M')}-{zone_code}"


def pick_top_resources(optimizer_result: Optional[dict], fallback_resources: Optional[dict]) -> list:
    """Top 3 resource lines — prefers MILP-optimal counts, falls back to the lookup table."""
    items = []
    if optimizer_result and str(optimizer_result.get("status", "")).startswith(("OPTIMAL", "FALLBACK")):
        for key, label in [
            ("inspector_strike_teams", "Inspector + 4-Constable Strike Team"),
            ("constable_teams_2p", "2-Constable Foot Patrol Team"),
            ("barricades_required", "Physical Barricade Unit"),
            ("signal_diversion_protocols", "Signal Diversion Activation"),
        ]:
            n = optimizer_result.get(key, 0)
            if n and n > 0:
                items.append(f"{int(n)} × {label}")
    if len(items) < 3 and fallback_resources:
        for key, label in [
            ("officers", "Traffic Constables"),
            ("home_guards", "Home Guard Personnel"),
            ("cctv_teams", "Mobile CCTV Team"),
            ("tow_trucks", "Tow Truck / Crane"),
            ("traffic_marshals", "Traffic Marshal (crowd flow)"),
        ]:
            n = fallback_resources.get(key, 0)
            if n and n > 0 and len(items) < 3:
                items.append(f"{int(n)} × {label}")
    return items[:3] if items else ["1 × Traffic Marshal Point (standard minimum deployment)"]


def build_barricade_lines(deployments: list, max_items: int = 5) -> list:
    """Plain-language barricade positions, ordered by priority — for a constable, not a GIS analyst."""
    if not deployments:
        return ["No barricades required for this severity level."]
    order = {"EMERGENCY": 0, "CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}
    ranked = sorted(
        [d for d in deployments if d.get("type") != "MARSHAL_POINT" or "Emergency" not in d.get("road", "")],
        key=lambda d: order.get(d.get("priority", "MEDIUM"), 5),
    )
    lines = []
    for d in ranked[:max_items]:
        lines.append(f"{d['id']} — {d['road']}: {d['instruction']}")
    return lines or ["No barricades required for this severity level."]


def build_emergency_corridor_line(deployments: list) -> str:
    for d in deployments or []:
        if d.get("id") == "EC01" or "Emergency Corridor" in d.get("road", ""):
            return f"Keep {d['road'].replace(' (Keep OPEN)', '')} OPEN at all times for ambulance, fire engine, and police vehicles. Do NOT place any barricade here."
    return "No dedicated emergency corridor flagged for this severity — maintain standard ambulance access on the widest approach road."


def build_diversion_lines(routes: list) -> list:
    if not routes:
        return ["No alternate route computed — use officer judgement on nearest open road."]
    lines = []
    for i, r in enumerate(routes, start=1):
        via = ", ".join(r.get("via_roads", [])[:3]) or "local roads"
        lines.append(f"Route {i}: {r['length_km']} km (~{r['eta_min']:.0f} min) via {via}")
    return lines


# ─── HTML card (in-app preview) ────────────────────────────────────────────────
def generate_dispatch_html(
    dispatch_id: str, event_cause: str, zone: str, location_label: str,
    lat: float, lng: float, event_dt: datetime.datetime, deploy_at: datetime.datetime,
    severity: str, confidence_pct: float, expected_clearance_mins: float,
    resource_lines: list, barricade_lines: list, emergency_corridor_line: str,
    diversion_lines: list, generated_at: datetime.datetime,
) -> str:
    sev_color = SEVERITY_COLORS_HEX.get(severity, "#6B7280")
    resources_html = "".join(f"<li>{r}</li>" for r in resource_lines)
    barricades_html = "".join(f"<li>{b}</li>" for b in barricade_lines)
    diversion_html = "".join(f"<li>{d}</li>" for d in diversion_lines)

    return f"""
    <div style="background:#FFFFFF;border:2px solid #111827;border-radius:14px;
                padding:0;overflow:hidden;font-family:Inter,Arial,sans-serif;max-width:720px">
        <div style="background:#1D4ED8;color:#FFFFFF;padding:16px 22px;display:flex;
                    justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
            <div>
                <div style="font-size:11px;letter-spacing:1px;opacity:0.85;font-weight:600">AEGISFLOW · ASTRAM SDK EXTENSION</div>
                <div style="font-size:19px;font-weight:800;margin-top:2px">FIELD DISPATCH ORDER</div>
            </div>
            <div style="text-align:right">
                <div style="font-size:11px;opacity:0.85">DISPATCH ID</div>
                <div style="font-size:15px;font-weight:700">{dispatch_id}</div>
            </div>
        </div>
        <div style="padding:18px 22px">
            <div style="display:inline-block;background:{sev_color}1A;border:1px solid {sev_color};
                        color:{sev_color};border-radius:8px;padding:5px 12px;font-weight:700;
                        font-size:13px;margin-bottom:14px">
                {severity} SEVERITY &nbsp;·&nbsp; {confidence_pct:.0f}% confidence
            </div>

            <table style="width:100%;font-size:13px;color:#111827;border-collapse:collapse;margin-bottom:14px">
                <tr><td style="padding:3px 0;color:#6B7280;width:38%">Event Type</td><td><b>{event_cause.replace('_',' ').title()}</b></td></tr>
                <tr><td style="padding:3px 0;color:#6B7280">Zone</td><td><b>{zone}</b></td></tr>
                <tr><td style="padding:3px 0;color:#6B7280">Location</td><td><b>{location_label}</b> &nbsp;<span style="color:#9CA3AF">({lat:.4f}, {lng:.4f})</span></td></tr>
                <tr><td style="padding:3px 0;color:#6B7280">Event Start</td><td><b>{event_dt.strftime('%d %b %Y, %I:%M %p')}</b></td></tr>
                <tr><td style="padding:3px 0;color:#6B7280">Deploy By</td><td><b style="color:{sev_color}">{deploy_at.strftime('%d %b %Y, %I:%M %p')}</b></td></tr>
                <tr><td style="padding:3px 0;color:#6B7280">Expected Clearance</td><td><b>{expected_clearance_mins:.0f} min</b></td></tr>
            </table>

            <div style="font-size:12px;font-weight:700;letter-spacing:0.5px;color:#374151;margin-bottom:4px">RESOURCES TO DEPLOY</div>
            <ul style="margin:0 0 14px 18px;font-size:13px;color:#111827;line-height:1.6">{resources_html}</ul>

            <div style="font-size:12px;font-weight:700;letter-spacing:0.5px;color:#374151;margin-bottom:4px">BARRICADE POSITIONS</div>
            <ol style="margin:0 0 14px 18px;font-size:13px;color:#111827;line-height:1.6">{barricades_html}</ol>

            <div style="background:#FEF2F2;border:1px solid #FECACA;border-left:4px solid #EF4444;
                        border-radius:8px;padding:10px 12px;margin-bottom:14px;font-size:13px;color:#7F1D1D">
                🚑 <b>EMERGENCY CORRIDOR:</b> {emergency_corridor_line}
            </div>

            <div style="font-size:12px;font-weight:700;letter-spacing:0.5px;color:#374151;margin-bottom:4px">ALTERNATE DIVERSION ROUTES</div>
            <ul style="margin:0 0 4px 18px;font-size:13px;color:#111827;line-height:1.6">{diversion_html}</ul>

            <div style="border-top:1px solid #E5E7EB;margin-top:16px;padding-top:10px;
                        font-size:11px;color:#9CA3AF;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px">
                <span>Generated by TraffiX on {generated_at.strftime('%d %b %Y, %I:%M %p')}</span>
                <span>Decision support only — Sub-Inspector judgement on the ground takes precedence.</span>
            </div>
        </div>
    </div>
    """


# ─── PDF card (print-ready, downloadable) ──────────────────────────────────────
def generate_dispatch_pdf(
    dispatch_id: str, event_cause: str, zone: str, location_label: str,
    lat: float, lng: float, event_dt: datetime.datetime, deploy_at: datetime.datetime,
    severity: str, confidence_pct: float, expected_clearance_mins: float,
    resource_lines: list, barricade_lines: list, emergency_corridor_line: str,
    diversion_lines: list, generated_at: datetime.datetime,
) -> bytes:
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_margins(14, 14, 14)

    sev_rgb = SEVERITY_COLORS.get(severity, (107, 114, 128))

    # ── Header band ──
    pdf.set_fill_color(29, 78, 216)  # #1D4ED8
    pdf.rect(0, 0, 210, 24, style="F")
    pdf.set_xy(14, 5)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, "AEGISFLOW \u00b7 ASTRAM SDK EXTENSION", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(14)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, "FIELD DISPATCH ORDER", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(140, 6)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(56, 4, "DISPATCH ID", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(140, 11)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(56, 5, dispatch_id, align="R")

    pdf.set_y(30)
    pdf.set_text_color(17, 24, 39)

    # ── Severity badge ──
    pdf.set_fill_color(*sev_rgb)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    badge_txt = f"  {severity} SEVERITY  ({confidence_pct:.0f}% confidence)  "
    badge_w = pdf.get_string_width(badge_txt) + 4
    pdf.cell(badge_w, 8, badge_txt, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Event details table ──
    pdf.set_text_color(17, 24, 39)
    rows = [
        ("Event Type", event_cause.replace("_", " ").title()),
        ("Zone", zone),
        ("Location", f"{location_label}  ({lat:.4f}, {lng:.4f})"),
        ("Event Start", event_dt.strftime("%d %b %Y, %I:%M %p")),
        ("Deploy By", deploy_at.strftime("%d %b %Y, %I:%M %p")),
        ("Expected Clearance", f"{expected_clearance_mins:.0f} min"),
    ]
    for label, value in rows:
        pdf.set_x(14)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(107, 114, 128)
        pdf.cell(45, 6.5, label)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(17, 24, 39)
        pdf.multi_cell(0, 6.5, _pdf_safe(value))
    pdf.ln(3)

    def section_header(title: str):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(55, 65, 81)
        pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")

    def numbered_list(items: list):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(17, 24, 39)
        for i, item in enumerate(items, start=1):
            pdf.set_x(16)
            pdf.multi_cell(0, 6, _pdf_safe(f"{i}. {item}"))
        pdf.ln(2)

    section_header("RESOURCES TO DEPLOY")
    numbered_list(resource_lines)

    section_header("BARRICADE POSITIONS")
    numbered_list(barricade_lines)

    # ── Emergency corridor box ──
    section_header("EMERGENCY CORRIDOR")
    pdf.set_fill_color(254, 242, 242)
    pdf.set_draw_color(239, 68, 68)
    pdf.set_text_color(127, 29, 29)
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 6, _pdf_safe("  " + emergency_corridor_line), fill=True, border=1)
    pdf.ln(3)

    section_header("ALTERNATE DIVERSION ROUTES")
    numbered_list(diversion_lines)

    # ── Signature lines ──
    pdf.ln(4)
    pdf.set_draw_color(209, 213, 219)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(107, 114, 128)
    y = pdf.get_y()
    pdf.line(14, y + 8, 95, y + 8)
    pdf.line(115, y + 8, 196, y + 8)
    pdf.set_xy(14, y + 9)
    pdf.cell(81, 5, "Issued by (Sub-Inspector)")
    pdf.set_xy(115, y + 9)
    pdf.cell(81, 5, "Received by (Constable / HC)")

    # ── Footer ──
    pdf.set_auto_page_break(False)
    pdf.set_xy(14, 279)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(156, 163, 175)
    pdf.cell(0, 5, f"Generated by TraffiX on {generated_at.strftime('%d %b %Y, %I:%M %p')}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(14)
    pdf.cell(0, 5, "Decision support only - Sub-Inspector judgement on the ground takes precedence.")

    out = pdf.output()
    return bytes(out)
