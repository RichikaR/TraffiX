"""
TraffiX Traffic Tactical Optimizer
Mixed-Integer Linear Programming (MILP) for optimal resource deployment.

Cost coefficients are illustrative estimates modelled on typical Karnataka Police
duty allowances (7th Pay Commission scale) and BBMP road safety logistics costs.
They have NOT been independently verified against specific government resolutions
or SOPs. Teams using this system in production should validate these against
actual BTP/BBMP procurement records before treating them as authoritative.

Resource 0 — 2-Constable foot patrol team
  Cost index: 160  (illustrative)
  Basis: Estimated from Karnataka Police 7CPC Pay Matrix Level 3 (Constable
  ~₹25,500/month basic). Event-proportional index for a 2.4 hr average event.
  This is an approximation, not a verified procurement rate.

Resource 1 — Sector Inspector + 4 Constable strike team
  Cost index: 450  (illustrative)
  Basis: Estimated from 7CPC Pay Level 7 (Inspector) + 4 Constables.
  Event-proportional share of daily duty cost. Not verified against BTP
  deployment manifests or any specific HR circular.

Resource 2 — Physical barricade + placement crew
  Cost index: 95  (illustrative)
  Basis: Estimated range of ₹280–320 per barricade unit including transport
  and placement crew, normalised to the same index scale as personnel costs.
  Not verified against any specific BBMP standing committee resolution or
  rate card; treat as a reasonable planning estimate.

Resource 3 — Active signal diversion / manual override protocol
  Cost index: 580  (illustrative)
  Basis: Estimated cost of TMC operator overtime (4 hr) + junction marshal
  coordination. Not verified against any BTP TMC SOP or published tariff.

Impact mitigation values (delay-minutes absorbed per unit deployed):
  Constable team:  2.5 min — reduces queue formation, not blockage clearance
  Inspector team:  6.0 min — authoritative diversion + faster clearance decisions
  Barricade:       1.2 min — physical channelling reduces lane-merge delay
  Signal override: 10.0 min — eliminates signal cycle wait at affected junction
  Basis: Calibrated against ASTRAM clearance_mins distribution by response type.
  These are modelled estimates, not measured ground-truth values.
"""

import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

# Severity class mapping from TraffiX prediction output
SEVERITY_TO_OPTIMIZER = {
    "CRITICAL": "FULL_BLOCK",
    "HIGH":     "PARTIAL_BLOCK",
    "MODERATE": "NORMAL_OBSTRUCTION",
    "LOW":      "NORMAL_OBSTRUCTION",
}


class TrafficTacticalOptimizer:
    def __init__(self):
        # Resource cost indices (see module docstring for sourcing)
        # Index: [2-constable team, Inspector+4 team, barricade unit, signal override]
        self.resource_costs = np.array([160, 450, 95, 580])

        # Delay minutes absorbed per unit of each resource
        self.impact_mitigation = np.array([2.5, 6.0, 1.2, 10.0])

    def optimize_deployment(
        self,
        predicted_delay_mins: float,
        road_lanes: int,
        severity_class: str,
    ) -> dict:
        """
        Minimize total deployment cost while guaranteeing:
            sum(mitigation_i * x_i) >= predicted_delay_mins

        Decision variables (all integers):
          x[0]: number of 2-constable foot teams
          x[1]: number of Inspector strike teams
          x[2]: number of barricade units
          x[3]: signal diversion activations (0 or 1 on narrow roads, max 1 on 3-lane+)
        """
        num_resources = len(self.resource_costs)

        max_barricades = 15 if severity_class == "FULL_BLOCK" else 5
        # Signal override only makes sense on roads with ≥3 lanes
        max_signals = 1 if road_lanes >= 3 else 0

        bounds = Bounds(
            lb=[0, 0, 0, 0],
            ub=[4, 2, max_barricades, max_signals],
        )

        # Constraint: mitigation coverage must meet or exceed predicted delay
        constraints = LinearConstraint(
            np.array([-self.impact_mitigation]),
            lb=[-np.inf],
            ub=[-predicted_delay_mins],
        )

        integrality = np.ones(num_resources)  # all decision variables are integers

        res = milp(
            c=self.resource_costs,
            bounds=bounds,
            constraints=constraints,
            integrality=integrality,
        )

        if res.success:
            x = np.round(res.x)
            return {
                "status": "OPTIMAL",
                "constable_teams_2p":          int(x[0]),
                "inspector_strike_teams":       int(x[1]),
                "barricades_required":          int(x[2]),
                "signal_diversion_protocols":   int(x[3]),
                "total_mitigation_score":       float(np.dot(x, self.impact_mitigation)),
                "estimated_cost_index":         int(res.fun),
                # Explain the cost index to the operator
                "cost_breakdown": {
                    "constable_teams":   int(x[0] * self.resource_costs[0]),
                    "inspector_teams":   int(x[1] * self.resource_costs[1]),
                    "barricades":        int(x[2] * self.resource_costs[2]),
                    "signal_override":   int(x[3] * self.resource_costs[3]),
                    "note": (
                        "Cost index is illustrative — modelled on typical Karnataka "
                        "Police duty allowances (7CPC scale) + estimated BBMP "
                        "logistics costs. Not verified against official resolutions. "
                        "See optimizer.py docstring for full caveats."
                    ),
                },
            }
        else:
            return self._generate_fallback_manifest(severity_class)

    def _generate_fallback_manifest(self, severity_class: str) -> dict:
        """Used when MILP solver cannot find a feasible solution (rare)."""
        if severity_class == "FULL_BLOCK":
            return {
                "status": "FALLBACK_MAX_DEPLOYMENT",
                "constable_teams_2p": 2,
                "inspector_strike_teams": 1,
                "barricades_required": 10,
                "signal_diversion_protocols": 1,
                "total_mitigation_score": 22.2,
                "estimated_cost_index": 1445,
                "cost_breakdown": {},
            }
        return {
            "status": "FALLBACK_STANDARD",
            "constable_teams_2p": 1,
            "inspector_strike_teams": 0,
            "barricades_required": 3,
            "signal_diversion_protocols": 0,
            "total_mitigation_score": 3.7,
            "estimated_cost_index": 445,
            "cost_breakdown": {},
        }


if __name__ == "__main__":
    optimizer = TrafficTacticalOptimizer()
    result = optimizer.optimize_deployment(
        predicted_delay_mins=14.5, road_lanes=3, severity_class="FULL_BLOCK"
    )
    import json
    print("\n=== MILP Optimizer Test Output ===")
    print(json.dumps(result, indent=2))
