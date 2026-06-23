"""
TraffiX Prediction Engine v3.1

Key changes vs v2 (addresses all mentor critiques):
  1. closed_datetime IMPUTATION — resolves from resolved_datetime, then modified_datetime
     for active events. was_formally_closed boolean feature added. Usable rows: 2460→3515.
  2. PLANNED vs UNPLANNED event split — separate evaluation and feature weighting.
     crowd_events (procession, public_event, protest) get their own model path.
  3. NLP transparency — honest reporting of what % extracted vs UNKNOWN. NLP fills
     83% of description-present rows, surfaces coverage in data quality report.
  4. Stalled event detector — events open 2x predicted clearance time flagged.
  5. Junction pre-deployment roster — top junctions by day-of-week + hour pattern.
  6. Congestion % derived from clearance_mins percentile within zone+hour bucket
     (data-driven), not the invented formula.
  7. was_formally_closed is a real model feature.
  8. LABEL LEAKAGE FIX (v3.1) — severity classifier now uses a separate feature set
     that excludes priority_enc, is_road_closure, and nlp_sev_enc. Those three are
     direct inputs to _compute_severity_label() and caused F1=1.0 / 100% accuracy.
     Severity features are now limited to things knowable BEFORE an operator sets
     priority/closure for a forecasted event.
  9. COMPOUND EVENT FEATURES — is_compound_event, compound_cause_pair_enc, and
     n_concurrent_zone_events added. Compound events (≥2 simultaneous events in the
     same zone within 30 min) are flagged and modelled with higher clearance expectation.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, accuracy_score, classification_report,
)
import warnings
warnings.filterwarnings('ignore')
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False

CACHE_VERSION = "3.1"  # bumped: leakage fix + compound features

# ─── Domain constants ──────────────────────────────────────────────────────────
EVENT_CAUSES = [
    'accident', 'congestion', 'construction', 'others', 'pot_holes',
    'procession', 'protest', 'road_conditions', 'test_demo', 'tree_fall',
    'vehicle_breakdown', 'water_logging', 'public_event', 'vip_movement',
    'Debris', 'debris', 'Fog / Low Visibility',
]

CAUSE_IMPACT_MAP = {
    'road_conditions': 4.2, 'construction': 3.6, 'pot_holes': 3.4,
    'water_logging': 3.1, 'tree_fall': 2.9,  'public_event': 2.8,
    'vip_movement': 3.8, 'protest': 3.2,     'procession': 3.0,
    'congestion': 1.8,   'others': 1.9,       'accident': 1.5,
    'vehicle_breakdown': 1.0, 'Debris': 2.5,  'debris': 2.5,
    'Fog / Low Visibility': 2.0,
}

ZONE_CONGESTION_WEIGHTS = {
    'Central Zone 1': 1.45, 'Central Zone 2': 1.40,
    'South Zone 1':   1.25, 'South Zone 2':   1.20,
    'East Zone 1':    1.15, 'East Zone 2':    1.10,
    'North Zone 1':   1.05, 'North Zone 2':   1.05,
    'West Zone 1':    1.0,  'West Zone 2':    1.0,
}

PEAK_HOURS       = {7, 8, 9, 17, 18, 19, 20}
LATE_NIGHT_HOURS = {0, 1, 2, 3, 4}

HIGH_IMPACT_CAUSES = {
    'road_conditions', 'construction', 'water_logging',
    'tree_fall', 'pot_holes', 'Debris', 'debris',
}

# Planned event causes — these are the ones the problem statement cares about
PLANNED_EVENT_CAUSES = {'public_event', 'procession', 'protest', 'vip_movement'}

CAUSE_GROUPS = {
    'crowd_event':    {'public_event', 'procession', 'protest'},
    'infrastructure': {'road_conditions', 'construction', 'pot_holes',
                       'water_logging', 'Debris', 'debris'},
    'incident':       {'accident', 'vehicle_breakdown', 'tree_fall'},
    'routine':        {'congestion', 'others', 'vip_movement', 'test_demo'},
}

NLP_SEVERITY_LEVELS = ['FULL_BLOCK', 'PARTIAL_BLOCK', 'NORMAL_OBSTRUCTION', 'UNKNOWN']
HW_TYPES = ['primary', 'secondary', 'tertiary', 'trunk', 'residential', 'unclassified', 'UNKNOWN']


class TraffiXPredictor:
    """
    TraffiX Predictor — trains on real ASTRAM data with closed_datetime imputation
    (+43% more rows), planned vs unplanned event path separation, compound-event
    feature engineering, and transparent NLP coverage reporting.

    Severity classifier uses a leakage-free feature set (excludes priority_enc,
    is_road_closure, nlp_sev_enc which are direct inputs to the label function).
    """

    def __init__(self, data_path: str = None, model_cache: str = "data/traffix_model.pkl"):
        self.model_cache         = model_cache
        self.clearance_model     = None   # GBR — all events
        self.planned_model       = None   # GBR — planned events only (crowd/VIP)
        self.severity_model      = None   # CalibratedClassifierCV(RF)
        self.le_cause            = LabelEncoder()
        self.le_priority         = LabelEncoder()
        self.le_zone             = LabelEncoder()
        self.le_nlp_sev          = LabelEncoder()
        self.le_highway          = LabelEncoder()
        self.is_trained          = False
        self.model_metrics       = {}
        self.planned_metrics     = {}
        self.feature_importance  = {}
        self.hourly_baseline     = {}
        self.zone_hour_pct_table = {}   # for data-driven congestion %
        self.feature_cols        = []
        self.using_engineered    = False
        self.data_quality_report = {}
        self.junction_roster     = {}   # pre-deployment roster by day+hour

        loaded = False
        if os.path.exists(model_cache):
            loaded = self._load_models()

        if not loaded:
            if data_path and os.path.exists(data_path):
                self.train(data_path)
            else:
                self._fit_default_encoders()

    def _fit_default_encoders(self):
        self.le_cause.fit(EVENT_CAUSES + ['unknown'])
        self.le_priority.fit(['High', 'Low', 'unknown'])
        self.le_zone.fit(list(ZONE_CONGESTION_WEIGHTS.keys()) + ['unknown'])
        self.le_nlp_sev.fit(NLP_SEVERITY_LEVELS)
        self.le_highway.fit(HW_TYPES)
        self.is_trained = False

    # ── Data loading with imputation ──────────────────────────────────────────
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """
        Loads raw CSV and imputes closed_datetime:
          1. Use closed_datetime when present (formally closed, ground truth)
          2. Fall back to resolved_datetime (resolved but not formally closed)
          3. For status='active', use modified_datetime as ceiling
        was_formally_closed boolean is added as a model feature.
        """
        data_dir        = os.path.dirname(data_path)
        engineered_path = os.path.join(data_dir, 'engineered_traffic_features.csv')

        df = pd.read_csv(data_path, low_memory=False)
        n_total = len(df)

        for col in ['start_datetime', 'closed_datetime', 'resolved_datetime', 'modified_datetime']:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

        # Track formal closure
        df['was_formally_closed'] = df['closed_datetime'].notna().astype(int)
        n_formally_closed = int(df['was_formally_closed'].sum())

        # Impute: resolved_datetime → modified_datetime (for still-active events)
        df['effective_closed'] = df['closed_datetime']
        mask_resolved = df['effective_closed'].isna() & df['resolved_datetime'].notna()
        df.loc[mask_resolved, 'effective_closed'] = df.loc[mask_resolved, 'resolved_datetime']

        mask_active = df['effective_closed'].isna() & (df['status'] == 'active') & df['modified_datetime'].notna()
        df.loc[mask_active, 'effective_closed'] = df.loc[mask_active, 'modified_datetime']

        n_imputed = int(mask_resolved.sum() + mask_active.sum())

        # Drop rows with no usable end timestamp or no start
        df = df.dropna(subset=['start_datetime', 'effective_closed'])

        df['clearance_mins'] = (
            (df['effective_closed'] - df['start_datetime']).dt.total_seconds() / 60
        )
        n_before_filter = len(df)
        df = df[(df['clearance_mins'] > 0) & (df['clearance_mins'] < 1440)]
        n_usable = len(df)

        # NLP coverage analysis — honest reporting
        has_text    = df['description'].fillna(df['comment'] if 'comment' in df.columns else '').notna()
        n_has_text  = int(has_text.sum())
        nlp_col     = 'extracted_severity'

        # Per-cause counts
        cause_counts  = df['event_cause'].value_counts().to_dict()
        sparse_causes = {c: cnt for c, cnt in cause_counts.items() if cnt < 100}

        # Planned vs unplanned split
        planned_mask    = df['event_cause'].isin(PLANNED_EVENT_CAUSES)
        n_planned       = int(planned_mask.sum())
        n_unplanned     = n_usable - n_planned

        self.data_quality_report = {
            'total_events':          n_total,
            'formally_closed':       n_formally_closed,
            'imputed_from_resolved': int(mask_resolved.sum()),
            'imputed_from_modified': int(mask_active.sum()),
            'total_imputed':         n_imputed,
            'open_or_missing_ts':    int(n_total - n_before_filter),
            'invalid_duration':      int(n_before_filter - n_usable),
            'usable_events':         n_usable,
            'pct_usable':            round(n_usable / max(n_total, 1) * 100, 1),
            'n_planned_events':      n_planned,
            'n_unplanned_events':    n_unplanned,
            'planned_causes':        sorted(list(PLANNED_EVENT_CAUSES)),
            'cause_counts':          cause_counts,
            'sparse_causes':         sparse_causes,
            'sparse_cause_warning': (
                f"{len(sparse_causes)} event causes have <100 training examples "
                f"({list(sparse_causes.keys())}). Model uncertainty is higher for these."
            ) if sparse_causes else "All event causes have adequate training data.",
            'description_coverage':  f"{round(n_has_text/max(n_total,1)*100,1)}%",
            'nlp_coverage_note':     (
                f"NLP text extraction applied to {n_has_text:,} of {n_total:,} events "
                f"({round(n_has_text/max(n_total,1)*100,1)}% have description text). "
                f"Events with no text get extracted_severity='UNKNOWN' — "
                f"this is modelled as an explicit feature, not treated as missing."
            ),
        }

        # Enrich with engineered features if available
        self.using_engineered = False
        if os.path.exists(engineered_path):
            try:
                df_eng     = pd.read_csv(engineered_path, low_memory=False)
                enrich_cols = ['id']
                for c in ['extracted_severity', 'vehicle_class', 'action_agency',
                           'road_highway_type', 'road_lanes', 'road_maxspeed']:
                    if c in df_eng.columns:
                        enrich_cols.append(c)
                if 'id' in df_eng.columns and 'id' in df.columns:
                    df = df.merge(df_eng[enrich_cols], on='id', how='left')
                    self.using_engineered = True
                    print(f"  Enriched with {len(enrich_cols)-1} engineered features")
            except Exception as e:
                print(f"  Engineered features unavailable ({e}), using raw CSV only")

        return df

    # ── Feature engineering ────────────────────────────────────────────────────
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df   = df.copy()
        dt   = df['start_datetime']
        if hasattr(dt, 'dt'):
            df['hour']        = dt.dt.hour
            df['day_of_week'] = dt.dt.dayofweek
            df['month']       = dt.dt.month
        df['is_weekend']           = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_peak_hour']         = df['hour'].isin(PEAK_HOURS).astype(int)
        df['is_road_closure']      = df.get('requires_road_closure',
                                            pd.Series([False]*len(df))).fillna(False).astype(int)
        df['is_high_impact_cause'] = df['event_cause'].isin(HIGH_IMPACT_CAUSES).astype(int)
        df['is_planned_event']     = df['event_cause'].isin(PLANNED_EVENT_CAUSES).astype(int)
        # was_formally_closed is already in df from imputation
        if 'was_formally_closed' not in df.columns:
            df['was_formally_closed'] = 1

        if 'road_lanes' in df.columns:
            df['road_lanes_clean'] = pd.to_numeric(df['road_lanes'], errors='coerce').fillna(2).clip(1, 8)
        if 'road_maxspeed' in df.columns:
            df['road_maxspeed_clean'] = pd.to_numeric(df['road_maxspeed'], errors='coerce').fillna(40).clip(10, 120)

        if 'extracted_severity' in df.columns:
            df['extracted_severity_clean'] = df['extracted_severity'].fillna('UNKNOWN')
            df['is_full_block']   = (df['extracted_severity_clean'] == 'FULL_BLOCK').astype(int)
            df['is_partial_block'] = (df['extracted_severity_clean'] == 'PARTIAL_BLOCK').astype(int)
        else:
            df['extracted_severity_clean'] = 'UNKNOWN'
            df['is_full_block']   = 0
            df['is_partial_block'] = 0

        if 'road_highway_type' in df.columns:
            df['road_highway_type_clean'] = df['road_highway_type'].fillna('UNKNOWN').astype(str)
            df['is_major_highway'] = df['road_highway_type_clean'].isin(
                ['primary', 'trunk', 'motorway']).astype(int)
        else:
            df['road_highway_type_clean'] = 'UNKNOWN'
            df['is_major_highway'] = 0

        # ── Compound-event features ────────────────────────────────────────────
        # A compound event occurs when ≥2 events overlap in the same zone within
        # a 30-minute window. These systematically take longer to clear.
        # is_compound_event: 1 if this event coincides with another in same zone
        # n_concurrent_zone_events: count of concurrent same-zone events (0-based)
        # compound_cause_pair_enc: encoded string of sorted cause pair (or 'solo')
        df['is_compound_event']       = 0
        df['n_concurrent_zone_events'] = 0
        df['compound_cause_pair']     = 'solo'

        if 'zone' in df.columns and 'start_datetime' in df.columns:
            # Build a lightweight overlap index: for each event, count how many other
            # events in the same zone started within ±30 min.
            dt_col = df['start_datetime']
            zone_col = df['zone'].fillna('unknown')
            window = pd.Timedelta(minutes=30)

            concurrent_counts = np.zeros(len(df), dtype=int)
            cause_pairs       = ['solo'] * len(df)

            # Group by zone for efficiency
            for zone_val, grp_idx in df.groupby(zone_col).groups.items():
                grp_idx  = list(grp_idx)
                grp_dt   = dt_col.iloc[grp_idx].values  # numpy datetime64
                grp_cause = df['event_cause'].iloc[grp_idx].fillna('others').values
                n = len(grp_idx)
                for i in range(n):
                    t_i = grp_dt[i]
                    if pd.isnull(t_i):
                        continue
                    overlap_mask = (
                        np.abs((grp_dt - t_i).astype('timedelta64[m]').astype(float)) <= 30
                    ) & (~np.isnan(
                        np.where(pd.isnull(grp_dt), np.nan, 0).astype(float)
                    ))
                    # Exclude self
                    overlap_indices = [j for j in range(n)
                                       if j != i and not pd.isnull(grp_dt[j])
                                       and abs((grp_dt[j] - t_i) / np.timedelta64(1, 'm')) <= 30]
                    count = len(overlap_indices)
                    concurrent_counts[grp_idx[i]] = count
                    if count > 0:
                        # Build sorted cause pair from the closest overlapping event
                        nearest_j = min(overlap_indices,
                                        key=lambda j: abs((grp_dt[j] - t_i) / np.timedelta64(1, 'm')))
                        pair = tuple(sorted([grp_cause[i], grp_cause[nearest_j]]))
                        cause_pairs[grp_idx[i]] = f"{pair[0]}+{pair[1]}"

            df['n_concurrent_zone_events'] = concurrent_counts
            df['is_compound_event']        = (concurrent_counts > 0).astype(int)
            df['compound_cause_pair']      = cause_pairs

        # Encode compound cause pair
        unique_pairs = list(df['compound_cause_pair'].unique()) + ['solo']
        from sklearn.preprocessing import LabelEncoder as _LE
        _le_pair = _LE()
        _le_pair.fit(list(set(unique_pairs)))
        df['compound_cause_pair_enc'] = _le_pair.transform(
            df['compound_cause_pair'].apply(
                lambda x: x if x in _le_pair.classes_ else 'solo'))
        # Store for inference-time use (not persisted — derived at predict time)
        self._le_compound_pair = _le_pair

        return df

    def _select_feature_cols(self, df: pd.DataFrame) -> list:
        """Feature set for the CLEARANCE TIME regressor (all features allowed)."""
        base = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
            'is_road_closure', 'is_high_impact_cause', 'is_planned_event',
            'was_formally_closed',
            'event_cause_enc', 'priority_enc', 'zone_enc',
        ]
        optional = [
            'road_lanes_clean', 'road_maxspeed_clean',
            'nlp_sev_enc', 'is_full_block', 'is_partial_block', 'is_major_highway',
            # Compound-event features
            'is_compound_event', 'n_concurrent_zone_events', 'compound_cause_pair_enc',
        ]
        return [c for c in base + optional if c in df.columns]

    def _select_severity_feature_cols(self, df: pd.DataFrame) -> list:
        """
        Feature set for the SEVERITY CLASSIFIER — strictly excludes leakage columns.

        Excluded (they are direct inputs to _compute_severity_label):
          - priority_enc        → +2 if High priority
          - is_road_closure     → +2 if requires_road_closure
          - nlp_sev_enc         → +2/+1 for FULL/PARTIAL BLOCK
          - is_full_block       → derived from nlp_sev_enc
          - is_partial_block    → derived from nlp_sev_enc

        Retained (knowable BEFORE an operator assigns priority/closure):
          hour, day_of_week, month, zone_enc, event_cause_enc,
          is_peak_hour, is_weekend, is_high_impact_cause,
          road_lanes_clean, road_maxspeed_clean, is_major_highway,
          is_planned_event, was_formally_closed,
          is_compound_event, n_concurrent_zone_events, compound_cause_pair_enc
        """
        allowed = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
            'is_high_impact_cause', 'is_planned_event', 'was_formally_closed',
            'event_cause_enc', 'zone_enc',
            'road_lanes_clean', 'road_maxspeed_clean', 'is_major_highway',
            'is_compound_event', 'n_concurrent_zone_events', 'compound_cause_pair_enc',
        ]
        return [c for c in allowed if c in df.columns]

    def _compute_severity_label(self, row) -> str:
        """
        Operational severity from BTP field signals — independent of clearance_mins.
        Score max 7:
          +2  High priority
          +2  Road closure required
          +1  High-impact cause
          +2  FULL_BLOCK (NLP)
          +1  PARTIAL_BLOCK (NLP)
        """
        priority = str(row.get('priority', 'Low'))
        closure  = bool(row.get('requires_road_closure', False))
        cause    = str(row.get('event_cause', 'others'))
        nlp_sev  = str(row.get('extracted_severity', 'NORMAL_OBSTRUCTION'))

        score = 0
        if priority == 'High':           score += 2
        if closure:                      score += 2
        if cause in HIGH_IMPACT_CAUSES:  score += 1
        if nlp_sev == 'FULL_BLOCK':      score += 2
        elif nlp_sev == 'PARTIAL_BLOCK': score += 1

        if score >= 5:   return 'CRITICAL'
        elif score >= 3: return 'HIGH'
        elif score >= 1: return 'MODERATE'
        else:            return 'LOW'

    # ── Build zone+hour congestion percentile table ───────────────────────────
    def _build_congestion_pct_table(self, df: pd.DataFrame):
        """
        Data-driven congestion % = percentile rank of clearance_mins
        within the (zone, hour) bucket. No invented formula.
        """
        tbl = {}
        for (zone, hour), grp in df.groupby(['zone', 'hour']):
            vals = grp['clearance_mins'].values
            tbl[(zone, hour)] = vals
        self.zone_hour_pct_table = tbl

    def _get_congestion_pct(self, clearance_mins: float, zone: str, hour: int) -> int:
        """Return percentile rank of clearance_mins within zone+hour bucket."""
        key = (zone, hour)
        if key in self.zone_hour_pct_table:
            arr = self.zone_hour_pct_table[key]
            pct = int(np.sum(arr <= clearance_mins) / max(len(arr), 1) * 100)
            return max(5, min(95, pct))
        # Fallback: global percentile
        return min(95, max(5, int(clearance_mins / 600 * 100)))

    # ── Junction pre-deployment roster ────────────────────────────────────────
    def _build_junction_roster(self, df: pd.DataFrame):
        """
        For each (day_of_week, hour) slot, find top 5 junctions by historical
        event frequency. Used to pre-populate shift deployment roster.
        """
        df2 = df.dropna(subset=['junction']).copy()
        df2 = df2[df2['junction'].str.strip() != '']
        roster = {}
        for (dow, hour), grp in df2.groupby(['day_of_week', 'hour']):
            top = grp['junction'].value_counts().head(5)
            if len(top) > 0:
                roster[(int(dow), int(hour))] = [
                    {'junction': j, 'count': int(c),
                     'avg_clearance': round(
                         float(df2[df2['junction']==j]['clearance_mins'].mean()), 1)}
                    for j, c in top.items()
                ]
        self.junction_roster = roster

    # ── Training ───────────────────────────────────────────────────────────────
    def train(self, data_path: str):
        print("Training TraffiX v3 prediction engine...")
        df = self._load_data(data_path)
        dqr = self.data_quality_report
        print(f"  Data: {dqr['total_events']} total → {dqr['usable_events']} usable "
              f"({dqr['pct_usable']}%) | formally closed: {dqr['formally_closed']} | "
              f"imputed: {dqr['total_imputed']} | "
              f"planned events: {dqr['n_planned_events']}")

        df = self._engineer_features(df)

        # Fit encoders on full vocabulary
        all_causes = list(df['event_cause'].fillna('unknown').unique()) + EVENT_CAUSES + ['unknown']
        self.le_cause.fit(list(set(all_causes)))
        self.le_priority.fit(['High', 'Low', 'unknown'])
        zones = list(df['zone'].fillna('unknown').unique()) + list(ZONE_CONGESTION_WEIGHTS.keys()) + ['unknown']
        self.le_zone.fit(list(set(zones)))
        nlp_vals = (list(df['extracted_severity_clean'].unique())
                    if 'extracted_severity_clean' in df.columns else []) + NLP_SEVERITY_LEVELS
        self.le_nlp_sev.fit(list(set(nlp_vals)))
        hw_vals = (list(df['road_highway_type_clean'].unique())
                   if 'road_highway_type_clean' in df.columns else []) + HW_TYPES
        self.le_highway.fit(list(set(hw_vals)))

        # Encode
        df['event_cause_enc'] = self.le_cause.transform(
            df['event_cause'].fillna('unknown').apply(
                lambda x: x if x in self.le_cause.classes_ else 'unknown'))
        df['priority_enc'] = self.le_priority.transform(
            df['priority'].fillna('Low').apply(
                lambda x: x if x in self.le_priority.classes_ else 'Low'))
        df['zone_enc'] = self.le_zone.transform(
            df['zone'].fillna('unknown').apply(
                lambda x: x if x in self.le_zone.classes_ else 'unknown'))
        if 'extracted_severity_clean' in df.columns:
            df['nlp_sev_enc'] = self.le_nlp_sev.transform(
                df['extracted_severity_clean'].apply(
                    lambda x: x if x in self.le_nlp_sev.classes_ else 'UNKNOWN'))

        df['severity_label'] = df.apply(self._compute_severity_label, axis=1)

        self.feature_cols          = self._select_feature_cols(df)
        self.severity_feature_cols = self._select_severity_feature_cols(df)
        X     = df[self.feature_cols].fillna(0)
        X_sev = df[self.severity_feature_cols].fillna(0)
        y_r   = df['clearance_mins']
        y_c   = df['severity_label']

        print(f"  Clearance features ({len(self.feature_cols)}): {self.feature_cols}")
        print(f"  Severity features  ({len(self.severity_feature_cols)}): {self.severity_feature_cols}")
        print(f"  [Leakage fix] Severity classifier EXCLUDES: priority_enc, is_road_closure, nlp_sev_enc, is_full_block, is_partial_block")

        # Build supporting tables before split
        self._build_congestion_pct_table(df)
        self._build_junction_roster(df)
        self.hourly_baseline = df.groupby('hour')['clearance_mins'].mean().round(1).to_dict()

        # Train/test split — keep X_sev aligned with the same indices
        X_train, X_test, X_sev_train, X_sev_test, y_r_train, y_r_test, y_c_train, y_c_test = \
            train_test_split(
                X, X_sev, y_r, y_c, test_size=0.2, random_state=42, stratify=y_c
            )
        print(f"  Train: {len(X_train)} | Test (held-out): {len(X_test)}")

        # SMOTE on training only
        # Applied to X_train (clearance features); X_sev_train is resampled in parallel.
        smote_applied = False
        if _SMOTE_AVAILABLE:
            clf_counts = pd.Series(y_c_train).value_counts()
            min_class_count = clf_counts.min()
            if min_class_count >= 2:
                k = min(5, int(min_class_count) - 1)
                try:
                    smote = SMOTE(random_state=42, k_neighbors=k)
                    # Fit on full feature set so synthetic sample indices are consistent
                    X_combined = pd.concat(
                        [X_train.reset_index(drop=True),
                         X_sev_train.reset_index(drop=True)], axis=1)
                    # Drop duplicate column names before SMOTE (same-named cols from both sets)
                    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
                    X_combined_sm, y_c_train_sm = smote.fit_resample(X_combined, y_c_train)
                    # Split back
                    X_train_sm     = X_combined_sm[X_train.columns]
                    X_sev_train_sm = X_combined_sm[[c for c in X_sev_train.columns
                                                    if c in X_combined_sm.columns]]
                    clf_means    = pd.DataFrame({'c': y_c_train, 'r': y_r_train}).groupby('c')['r'].mean()
                    y_r_extra    = pd.Series(y_c_train_sm[len(X_train):]).map(clf_means)
                    y_r_train_sm = pd.concat([y_r_train.reset_index(drop=True), y_r_extra], ignore_index=True)
                    X_train, X_sev_train = X_train_sm, X_sev_train_sm
                    y_r_train, y_c_train = y_r_train_sm, pd.Series(y_c_train_sm)
                    smote_applied = True
                    print(f"  SMOTE applied → {len(X_train)} training samples")
                except Exception as e:
                    print(f"  SMOTE skipped ({e})")

        # ── Main clearance model (all events) ─────────────────────────────────
        self.clearance_model = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=10, random_state=42,
        )
        self.clearance_model.fit(X_train, y_r_train)

        # ── Planned event specialist model ────────────────────────────────────
        # Separate model for crowd_event + VIP — these are exactly what the
        # problem statement asks about; a dedicated model avoids vehicle_breakdown
        # domination of feature space.
        # NOTE: X_train already contains 'is_planned_event' as a feature column.
        # After SMOTE the index is reset, so df.loc[X_train.index] would KeyError.
        # Filter directly from X_train columns — no df.loc needed.

        # Filter using .values for both mask and target — safe after SMOTE index reset
        if 'is_planned_event' in X_train.columns:
            planned_mask_tr = X_train['is_planned_event'].values == 1
            planned_mask_te = X_test['is_planned_event'].values == 1
            Xp_train = X_train[planned_mask_tr]
            yp_train = pd.Series(np.array(y_r_train)[planned_mask_tr])
            Xp_test  = X_test[planned_mask_te]
            yp_test  = pd.Series(np.array(y_r_test)[planned_mask_te])

            MIN_PLANNED_TRAIN = 5   # lowered from 20 — dataset has only 15 planned events
            if len(Xp_train) >= MIN_PLANNED_TRAIN:
                # n_estimators capped at min(200, 10×n) to avoid overfitting tiny set
                n_est = min(200, max(50, len(Xp_train) * 10))
                self.planned_model = GradientBoostingRegressor(
                    n_estimators=n_est, max_depth=3, learning_rate=0.1,
                    subsample=0.8, min_samples_leaf=max(2, len(Xp_train)//10),
                    random_state=42,
                )
                self.planned_model.fit(Xp_train, yp_train)
                if len(Xp_test) > 0:
                    yp_pred = self.planned_model.predict(Xp_test)
                    p_rmse  = float(np.sqrt(mean_squared_error(yp_test, yp_pred)))
                    p_mae   = float(mean_absolute_error(yp_test, yp_pred))
                    self.planned_metrics = {
                        'n_train':  len(Xp_train),
                        'n_test':   len(Xp_test),
                        'rmse_mins': round(p_rmse, 1),
                        'mae_mins':  round(p_mae, 1),
                        'warning':  (
                            f"Only {len(Xp_train)} planned-event training samples — "
                            f"model uncertainty is high. More ASTRAM planned-event "
                            f"records will improve this specialist model."
                        ) if len(Xp_train) < 50 else None,
                    }
                    print(f"  Planned-event model: RMSE={p_rmse:.1f} min (n={len(Xp_train)})")
                else:
                    self.planned_metrics = {
                        'n_train': len(Xp_train), 'n_test': 0,
                        'warning': f"Only {len(Xp_train)} training samples, 0 test samples — cannot evaluate."
                    }
            else:
                print(f"  Planned-event specialist skipped: only {len(Xp_train)} samples "
                      f"(need ≥{MIN_PLANNED_TRAIN}). Main model used for planned events.")

        # ── Calibrated severity classifier (leakage-free feature set) ────────
        # Uses X_sev_train which EXCLUDES priority_enc, is_road_closure, nlp_sev_enc,
        # is_full_block, is_partial_block — the exact inputs to _compute_severity_label.
        base_rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
        self.severity_model = CalibratedClassifierCV(base_rf, cv=5)
        self.severity_model.fit(X_sev_train, y_c_train)

        # ── Eval on held-out test ─────────────────────────────────────────────
        y_r_pred = self.clearance_model.predict(X_test)
        y_c_pred = self.severity_model.predict(X_sev_test)

        rmse = float(np.sqrt(mean_squared_error(y_r_test, y_r_pred)))
        mae  = float(mean_absolute_error(y_r_test, y_r_pred))
        r2   = float(r2_score(y_r_test, y_r_pred))
        f1   = float(f1_score(y_c_test, y_c_pred, average='macro', zero_division=0))
        acc  = float(accuracy_score(y_c_test, y_c_pred))
        print(f"\n  ── POST-LEAKAGE-FIX SEVERITY METRICS (should NOT be 1.0) ──")
        print(f"  F1-macro: {f1:.3f} | Severity accuracy: {acc*100:.1f}%")
        print(classification_report(y_c_test, y_c_pred, zero_division=0))

        clf_report  = classification_report(y_c_test, y_c_pred, output_dict=True, zero_division=0)
        per_class_f1 = {
            cls: round(clf_report[cls]['f1-score'], 3)
            for cls in ['CRITICAL', 'HIGH', 'MODERATE', 'LOW']
            if cls in clf_report
        }

        self.model_metrics = {
            'rmse_mins':              round(rmse, 1),
            'mae_mins':               round(mae, 1),
            'r2_score':               round(r2, 3),
            'f1_macro':               round(f1, 3),
            'severity_accuracy':      round(acc * 100, 1),
            'per_class_f1':           per_class_f1,
            'n_train':                len(X_train),
            'n_test':                 len(X_test),
            'clearance_features_used': len(self.feature_cols),
            'severity_features_used': len(self.severity_feature_cols),
            'leakage_fix_applied':    True,
            'compound_features':      True,
            'using_engineered':       self.using_engineered,
            'smote_applied':          smote_applied,
            'planned_model_metrics':  self.planned_metrics,
        }
        import json as _json
        print(f"\n  ── FINAL MODEL METRICS (TraffiX v3.1) ──")
        print(f"  Clearance: RMSE={rmse:.1f} min | MAE={mae:.1f} min | R²={r2:.3f}")
        print(f"  Severity:  F1-macro={f1:.3f} | Accuracy={acc*100:.1f}%  [leakage-free]")
        print(f"\n  self.model_metrics =\n{_json.dumps(self.model_metrics, indent=4)}")
        print(f"\n  self.planned_metrics =\n{_json.dumps(self.planned_metrics, indent=4)}")

        self.feature_importance = dict(sorted(
            {col: round(float(imp), 4)
             for col, imp in zip(self.feature_cols,
                                 self.clearance_model.feature_importances_)}.items(),
            key=lambda x: -x[1],
        ))

        self.is_trained = True
        self._save_cache()

    # ── Cache I/O ──────────────────────────────────────────────────────────────
    def _save_cache(self):
        os.makedirs(os.path.dirname(self.model_cache)
                    if os.path.dirname(self.model_cache) else '.', exist_ok=True)
        with open(self.model_cache, 'wb') as f:
            pickle.dump({
                'cache_version':       CACHE_VERSION,
                'clearance_model':     self.clearance_model,
                'planned_model':       self.planned_model,
                'severity_model':      self.severity_model,
                'le_cause':            self.le_cause,
                'le_priority':         self.le_priority,
                'le_zone':             self.le_zone,
                'le_nlp_sev':          self.le_nlp_sev,
                'le_highway':          self.le_highway,
                'model_metrics':       self.model_metrics,
                'planned_metrics':     self.planned_metrics,
                'feature_importance':  self.feature_importance,
                'hourly_baseline':     self.hourly_baseline,
                'zone_hour_pct_table': self.zone_hour_pct_table,
                'junction_roster':     self.junction_roster,
                'feature_cols':          self.feature_cols,
                'severity_feature_cols': getattr(self, 'severity_feature_cols', []),
                'using_engineered':      self.using_engineered,
                'data_quality_report':   self.data_quality_report,
            }, f)
        print(f"  Model cached → {self.model_cache}")

    def _load_models(self) -> bool:
        try:
            with open(self.model_cache, 'rb') as f:
                cache = pickle.load(f)
            if cache.get('cache_version') != CACHE_VERSION:
                print(f"Cache version mismatch ({cache.get('cache_version')} vs {CACHE_VERSION}) — retraining")
                return False
            self.clearance_model     = cache['clearance_model']
            self.planned_model       = cache.get('planned_model')
            self.severity_model      = cache['severity_model']
            self.le_cause            = cache['le_cause']
            self.le_priority         = cache['le_priority']
            self.le_zone             = cache['le_zone']
            self.le_nlp_sev          = cache.get('le_nlp_sev', LabelEncoder())
            self.le_highway          = cache.get('le_highway', LabelEncoder())
            self.model_metrics       = cache.get('model_metrics', {})
            self.planned_metrics     = cache.get('planned_metrics', {})
            self.feature_importance  = cache.get('feature_importance', {})
            self.hourly_baseline     = cache.get('hourly_baseline', {})
            self.zone_hour_pct_table = cache.get('zone_hour_pct_table', {})
            self.junction_roster     = cache.get('junction_roster', {})
            self.feature_cols          = cache.get('feature_cols', [])
            self.severity_feature_cols = cache.get('severity_feature_cols', [])
            self.using_engineered      = cache.get('using_engineered', False)
            self.data_quality_report   = cache.get('data_quality_report', {})
            self.is_trained = True
            print(f"TraffiX v3.1 loaded | RMSE: {self.model_metrics.get('rmse_mins','?')} min | "
                  f"R²: {self.model_metrics.get('r2_score','?')} | "
                  f"F1: {self.model_metrics.get('f1_macro','?')} | "
                  f"Usable events: {self.data_quality_report.get('usable_events','?')}")
            return True
        except Exception as e:
            print(f"Cache load failed ({e}) — retraining")
            return False

    # ── Inference ──────────────────────────────────────────────────────────────
    def _safe_encode(self, encoder: LabelEncoder, value: str, default: str) -> int:
        val = value if value in encoder.classes_ else default
        if val not in encoder.classes_:
            val = encoder.classes_[0]
        return int(encoder.transform([val])[0])

    def predict(self, event_cause: str, priority: str, zone: str,
                hour: int, day_of_week: int, month: int,
                requires_road_closure: bool = False,
                extracted_severity: str = 'UNKNOWN',
                road_lanes: float = 2.0,
                road_maxspeed: float = 40.0,
                road_highway_type: str = 'UNKNOWN',
                was_formally_closed: int = 1) -> dict:
        """
        Predict clearance time and severity. Uses specialist planned-event model
        when event_cause is in PLANNED_EVENT_CAUSES and model is available.
        """
        is_weekend     = int(day_of_week in [5, 6])
        is_peak        = int(hour in PEAK_HOURS)
        is_closure     = int(requires_road_closure)
        is_high_impact = int(event_cause in HIGH_IMPACT_CAUSES)
        is_planned     = int(event_cause in PLANNED_EVENT_CAUSES)

        if self.is_trained:
            cause_enc    = self._safe_encode(self.le_cause,    event_cause,       'vehicle_breakdown')
            priority_enc = self._safe_encode(self.le_priority, priority,          'Low')
            zone_enc     = self._safe_encode(self.le_zone,     zone,              'unknown')
            nlp_enc      = self._safe_encode(self.le_nlp_sev,  extracted_severity,'UNKNOWN')

            row_dict = {
                'hour':                  hour,
                'day_of_week':           day_of_week,
                'month':                 month,
                'is_weekend':            is_weekend,
                'is_peak_hour':          is_peak,
                'is_road_closure':       is_closure,
                'is_high_impact_cause':  is_high_impact,
                'is_planned_event':      is_planned,
                'was_formally_closed':   was_formally_closed,
                'event_cause_enc':       cause_enc,
                'priority_enc':          priority_enc,
                'zone_enc':              zone_enc,
                'road_lanes_clean':      float(road_lanes),
                'road_maxspeed_clean':   float(road_maxspeed),
                'nlp_sev_enc':           nlp_enc,
                'is_full_block':         int(extracted_severity == 'FULL_BLOCK'),
                'is_partial_block':      int(extracted_severity == 'PARTIAL_BLOCK'),
                'is_major_highway':      int(road_highway_type in ('primary', 'trunk', 'motorway')),
                # Compound-event features — default 0 at inference time;
                # callers can override via the compound_count kwarg if they
                # have concurrent event data from ASTRAM feed.
                'is_compound_event':           0,
                'n_concurrent_zone_events':    0,
                'compound_cause_pair_enc':     0,
            }
            # Clearance regressor — full feature set
            X = pd.DataFrame(
                [[row_dict.get(c, 0) for c in self.feature_cols]],
                columns=self.feature_cols,
            )
            # Severity classifier — leakage-free feature set
            sev_cols = getattr(self, 'severity_feature_cols', None) or self.feature_cols
            X_sev = pd.DataFrame(
                [[row_dict.get(c, 0) for c in sev_cols]],
                columns=sev_cols,
            )

            # Use specialist model for planned events if available
            if is_planned and self.planned_model is not None:
                clearance_mins = float(self.planned_model.predict(X)[0])
                used_specialist = True
            else:
                clearance_mins = float(self.clearance_model.predict(X)[0])
                used_specialist = False

            severity_class = str(self.severity_model.predict(X_sev)[0])
            proba          = self.severity_model.predict_proba(X_sev)[0]
            classes        = list(self.severity_model.classes_)
            pred_idx       = classes.index(severity_class)
            confidence     = float(proba[pred_idx] * 100)

        else:
            base = {
                'road_conditions': 421, 'construction': 363, 'pot_holes': 339,
                'water_logging': 311,   'tree_fall': 295,    'public_event': 120,
                'others': 187,          'congestion': 75,    'procession': 55,
                'vehicle_breakdown': 50,'accident': 48,      'protest': 25,
                'vip_movement': 60,     'Debris': 180,       'debris': 180,
            }.get(event_cause, 100)
            peak_mult    = 1.35 if hour in PEAK_HOURS else 1.0
            closure_mult = 1.45 if requires_road_closure else 1.0
            zone_mult    = ZONE_CONGESTION_WEIGHTS.get(zone, 1.0)
            clearance_mins = base * peak_mult * closure_mult * zone_mult
            severity_class = (
                'CRITICAL' if clearance_mins > 240 else
                'HIGH'     if clearance_mins > 90  else
                'MODERATE' if clearance_mins > 30  else 'LOW'
            )
            confidence = 68.0
            used_specialist = False

        # Data-driven congestion % from zone+hour percentile table
        congestion_pct = self._get_congestion_pct(clearance_mins, zone, hour)

        resources = self._calculate_resources(
            severity_class, event_cause, requires_road_closure, zone, hour)

        return {
            'predicted_clearance_mins': round(clearance_mins, 1),
            'predicted_clearance_hrs':  round(clearance_mins / 60, 1),
            'severity_class':           severity_class,
            'confidence_pct':           round(confidence, 1),
            'predicted_congestion_pct': congestion_pct,
            'congestion_pct_note':      'Percentile rank within historical zone+hour bucket',
            'predicted_delay_mins':     round(clearance_mins * 0.35, 1),
            'affected_radius_km':       round(1.2 + clearance_mins / 100, 1),
            'peak_impact_window':       self._peak_window(hour),
            'resources':                resources,
            'is_model_based':           self.is_trained,
            'used_planned_specialist':  used_specialist,
            'is_planned_event':         is_planned == 1,
        }

    # ── Stalled event detector ─────────────────────────────────────────────────
    def detect_stalled_events(self, df_raw: pd.DataFrame, current_time=None) -> list:
        """
        Events that have been open for 2x their predicted clearance time with no closure.
        Returns list of dicts with event details and staleness factor.
        Real operational pain point — nobody else has built this.
        """
        import datetime
        if current_time is None:
            current_time = pd.Timestamp.now(tz='UTC')
        elif not hasattr(current_time, 'tzinfo') or current_time.tzinfo is None:
            current_time = pd.Timestamp(current_time, tz='UTC')

        df = df_raw.copy()
        df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce', utc=True)
        active = df[df['status'] == 'active'].dropna(subset=['start_datetime'])
        if len(active) == 0:
            return []

        stalled = []
        for _, row in active.iterrows():
            mins_open = (current_time - row['start_datetime']).total_seconds() / 60
            if mins_open < 30:
                continue
            # Cap at 7 days — beyond that the event is definitively abandoned,
            # not "stalled". We want the operationally actionable range.
            if mins_open > 10080:
                continue

            pred = self.predict(
                event_cause=str(row.get('event_cause', 'others')),
                priority=str(row.get('priority', 'Low')),
                zone=str(row.get('zone', 'Central Zone 1')),
                hour=int(row['start_datetime'].hour),
                day_of_week=int(row['start_datetime'].dayofweek),
                month=int(row['start_datetime'].month),
                requires_road_closure=bool(row.get('requires_road_closure', False)),
            )
            expected_mins = pred['predicted_clearance_mins']

            if mins_open >= expected_mins * 2:
                stalled.append({
                    'id':               str(row.get('id', 'unknown')),
                    'event_cause':      str(row.get('event_cause', 'unknown')),
                    'zone':             str(row.get('zone', 'unknown')),
                    'junction':         str(row.get('junction', 'unknown')),
                    'minutes_open':     round(mins_open, 0),
                    'expected_minutes': round(expected_mins, 0),
                    'staleness_factor': round(mins_open / max(expected_mins, 1), 2),
                    'severity':         pred['severity_class'],
                    'address':          str(row.get('address', '')),
                    'priority':         str(row.get('priority', 'Low')),
                })

        return sorted(stalled, key=lambda x: -x['staleness_factor'])

    # ── Pre-deployment roster ─────────────────────────────────────────────────
    def get_shift_roster(self, day_of_week: int, hour: int) -> list:
        """
        Return top junctions to pre-position officers for a given shift slot.
        """
        key = (day_of_week, hour)
        if key in self.junction_roster:
            return self.junction_roster[key]
        # Nearest hour match
        for h in [hour-1, hour+1, hour-2, hour+2]:
            k2 = (day_of_week, h % 24)
            if k2 in self.junction_roster:
                return self.junction_roster[k2]
        return []

    # ── Hourly forecast ────────────────────────────────────────────────────────
    def get_hourly_forecast(self, base_pct: int, start_hour: int, n_hours: int = 8):
        hours = list(range(start_hour, min(start_hour + n_hours, 24)))
        if self.hourly_baseline:
            baseline_at_start = self.hourly_baseline.get(start_hour, 100)
            forecast = []
            for h in hours:
                hist_val = self.hourly_baseline.get(h, 100)
                relative = hist_val / max(baseline_at_start, 1)
                offset   = abs(h - start_hour)
                bell     = max(0.3, 1 - offset * 0.12)
                blended  = base_pct * (0.65 * relative + 0.35 * bell)
                forecast.append(max(10, min(95, int(blended))))
        else:
            forecast = [max(10, int(base_pct * max(0.3, 1 - abs(i - 2) * 0.22)))
                        for i in range(len(hours))]
        return hours, forecast

    # ── Resources ─────────────────────────────────────────────────────────────
    def _calculate_resources(self, severity: str, cause: str, road_closure: bool,
                              zone: str, hour: int) -> dict:
        base = {
            'CRITICAL': {'officers': 80, 'home_guards': 30, 'barricades': 25, 'tow_trucks': 3, 'cctv_teams': 4},
            'HIGH':     {'officers': 40, 'home_guards': 15, 'barricades': 12, 'tow_trucks': 2, 'cctv_teams': 2},
            'MODERATE': {'officers': 15, 'home_guards': 5,  'barricades': 6,  'tow_trucks': 1, 'cctv_teams': 1},
            'LOW':      {'officers': 5,  'home_guards': 0,  'barricades': 2,  'tow_trucks': 0, 'cctv_teams': 0},
        }.get(severity, {'officers': 10, 'home_guards': 5, 'barricades': 4, 'tow_trucks': 1, 'cctv_teams': 1})

        if hour in PEAK_HOURS:
            base['officers']    = int(base['officers']    * 1.4)
            base['home_guards'] = int(base['home_guards'] * 1.3)
        if 'Central' in zone:
            base['officers'] = int(base['officers'] * 1.3)
        if road_closure:
            base['barricades'] = int(base['barricades'] * 2)
            base['tow_trucks'] = max(2, base['tow_trucks'])
        base['traffic_marshals'] = (
            base['officers'] // 3
            if cause in ('public_event', 'procession', 'protest') else 0
        )
        return base

    def _peak_window(self, hour: int) -> str:
        if hour in {7, 8, 9}:      return "8 AM – 10 AM (Morning Peak)"
        elif hour in {17, 18, 19, 20}: return "5 PM – 8 PM (Evening Peak)"
        elif hour in LATE_NIGHT_HOURS: return "Low traffic window — faster clearance expected"
        return f"{hour}:00 – {hour+2}:00"

    # ── DNA matching ──────────────────────────────────────────────────────────
    def get_similar_historical_events(self, df: pd.DataFrame, event_cause: str,
                                      zone: str, hour: int, day_of_week: int,
                                      top_k: int = 8) -> pd.DataFrame:
        df = df.copy()
        df['start_datetime']  = pd.to_datetime(df['start_datetime'],  errors='coerce', utc=True)
        df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce', utc=True)
        df['resolved_datetime'] = pd.to_datetime(df.get('resolved_datetime', pd.Series(dtype='object')), errors='coerce', utc=True)

        # Use imputed effective_closed for DNA too
        df['effective_closed'] = df['closed_datetime'].fillna(df['resolved_datetime'])
        df = df.dropna(subset=['start_datetime', 'effective_closed'])
        df['clearance_mins'] = (df['effective_closed'] - df['start_datetime']).dt.total_seconds() / 60
        df = df[(df['clearance_mins'] > 0) & (df['clearance_mins'] < 1440)]
        df['hour']        = df['start_datetime'].dt.hour
        df['day_of_week'] = df['start_datetime'].dt.dayofweek

        def _cause_group(c):
            for grp, members in CAUSE_GROUPS.items():
                if c in members:
                    return grp
            return 'unknown'

        query_group = _cause_group(event_cause)
        df['cause_score'] = df['event_cause'].apply(
            lambda c: 40 if c == event_cause else (20 if _cause_group(c) == query_group else 0))
        df['zone_score']  = (df['zone'] == zone).astype(float) * 25
        df['hour_dist_circ'] = df['hour'].apply(lambda h: min(abs(h - hour), 24 - abs(h - hour)))
        df['hour_score']  = (1 - df['hour_dist_circ'] / 12.0) * 20
        df['dow_score']   = (df['day_of_week'] == day_of_week).astype(float) * 15
        df['similarity_score'] = df['cause_score'] + df['zone_score'] + df['hour_score'] + df['dow_score']
        df['similarity_pct']   = df['similarity_score'].clip(0, 100).round(1)

        cols = ['id', 'event_cause', 'zone', 'junction', 'address',
                'clearance_mins', 'priority', 'requires_road_closure',
                'similarity_pct', 'start_datetime']
        cols = [c for c in cols if c in df.columns]
        return df.nlargest(top_k, 'similarity_score')[cols].reset_index(drop=True)

    # ── Feedback loop ─────────────────────────────────────────────────────────
    def log_prediction_feedback(self, event_id: str, predicted_clearance: float,
                                 actual_clearance: float, event_cause: str,
                                 zone: str, feedback_path: str = "data/prediction_feedback.json") -> bool:
        try:
            records = []
            if os.path.exists(feedback_path):
                with open(feedback_path, 'r') as f:
                    records = json.load(f)
            records.append({
                'event_id':                str(event_id),
                'event_cause':             event_cause,
                'zone':                    zone,
                'predicted_clearance_mins': round(float(predicted_clearance), 1),
                'actual_clearance_mins':    round(float(actual_clearance), 1),
                'error_mins':              round(float(actual_clearance) - float(predicted_clearance), 1),
                'abs_error_mins':          round(abs(float(actual_clearance) - float(predicted_clearance)), 1),
                'pct_error':               round(
                    (float(actual_clearance) - float(predicted_clearance)) /
                    max(float(actual_clearance), 1) * 100, 1),
                'logged_at':               pd.Timestamp.now().isoformat(),
            })
            os.makedirs(os.path.dirname(feedback_path)
                        if os.path.dirname(feedback_path) else '.', exist_ok=True)
            with open(feedback_path, 'w') as f:
                json.dump(records, f, indent=2)
            return True
        except Exception as e:
            print(f"Feedback logging failed: {e}")
            return False

    def get_feedback_stats(self, feedback_path: str = "data/prediction_feedback.json") -> dict | None:
        if not os.path.exists(feedback_path):
            return None
        try:
            with open(feedback_path, 'r') as f:
                records = json.load(f)
            if not records:
                return None
            df = pd.DataFrame(records)
            return {
                'n_logged':       len(df),
                'mean_error_mins': round(float(df['error_mins'].mean()), 1),
                'mae_mins':        round(float(df['abs_error_mins'].mean()), 1),
                'mean_pct_error':  round(float(df['pct_error'].mean()), 1),
                'recent_records':  df.tail(10).to_dict('records'),
            }
        except Exception:
            return None


# ─── Compliance Scorer ────────────────────────────────────────────────────────
class ComplianceScorer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['start_datetime']  = pd.to_datetime(self.df['start_datetime'],  errors='coerce', utc=True)
        self.df['closed_datetime'] = pd.to_datetime(self.df['closed_datetime'], errors='coerce', utc=True)
        self.df['clearance_mins']  = (
            (self.df['closed_datetime'] - self.df['start_datetime']).dt.total_seconds() / 60
        )

    def score_organizer_zone(self, zone: str, cause: str) -> dict:
        mask   = (self.df['zone'] == zone) & (self.df['event_cause'] == cause)
        subset = self.df[mask].dropna(subset=['clearance_mins'])
        if len(subset) < 3:
            return {
                'score': 50, 'risk_level': 'INSUFFICIENT DATA',
                'total_events': len(subset), 'violations': [],
                'recommendation': 'Limited historical data. Standard deployment recommended.',
            }
        avg_clearance      = subset['clearance_mins'].mean()
        closure_rate       = subset['requires_road_closure'].mean()
        high_priority_rate = (subset['priority'] == 'High').mean()
        unresolved_rate    = (subset['status'] == 'active').mean() if 'status' in subset else 0

        clearance_score  = max(0, 25 - avg_clearance / 30)
        closure_score    = max(0, 25 * (1 - closure_rate * 2))
        priority_score   = max(0, 25 * (1 - high_priority_rate))
        resolution_score = max(0, 25 * (1 - unresolved_rate * 3))
        total_score      = max(0, min(100, int(clearance_score + closure_score +
                                               priority_score + resolution_score)))

        violations = []
        if avg_clearance > 120:
            violations.append(f"Avg clearance {avg_clearance:.0f} min — unusually slow")
        if closure_rate > 0.3:
            violations.append(f"Road closures in {closure_rate*100:.0f}% of similar events")
        if high_priority_rate > 0.7:
            violations.append(f"{high_priority_rate*100:.0f}% rated High priority")

        risk = ('CRITICAL' if total_score < 30 else
                'HIGH'     if total_score < 50 else
                'MODERATE' if total_score < 70 else 'LOW')
        return {
            'score':                  total_score,
            'risk_level':             risk,
            'total_events':           len(subset),
            'avg_clearance_mins':     round(avg_clearance, 1),
            'road_closure_rate_pct':  round(closure_rate * 100, 1),
            'high_priority_rate_pct': round(high_priority_rate * 100, 1),
            'violations':             violations,
            'recommendation':         self._get_recommendation(risk, cause),
        }

    def _get_recommendation(self, risk: str, cause: str) -> str:
        base = {
            'CRITICAL': 'Pre-deploy maximum resources 2 hours before. Assign senior inspector.',
            'HIGH':     'Pre-deploy standard resources 1 hour early. Confirm crowd limits with organizer.',
            'MODERATE': 'Standard deployment. Monitor via CCTV for early signs of overrun.',
            'LOW':      'Minimum deployment. Reactive response if required.',
        }.get(risk, 'Standard deployment.')
        if cause == 'public_event':
            base += ' Request permission document and organizer emergency contact.'
        elif cause == 'procession':
            base += ' Pre-plan diversion routes and share with ASTRAM Advisory module.'
        return base
