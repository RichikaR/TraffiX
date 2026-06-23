import re
import pandas as pd

class TrafficNLPEngine:
    def __init__(self):
        self.severity_patterns = {
            'FULL_BLOCK': re.compile(r'(full block|completely blocked|no movement|ರಸ್ತೆ ಬಂದ್|ಪೂರ್ತಿ ಬಂದ್)', re.IGNORECASE),
            'PARTIAL_BLOCK': re.compile(r'(slow movement|one side|heavy traffic|ನಿಧಾನಗತಿ|ಸ್ಲೋ)', re.IGNORECASE),
        }
        
        self.vehicle_patterns = {
            'BMTC_BUS': re.compile(r'(bmtc|bus|ಬಸ್)', re.IGNORECASE),
            'HEAVY_GOODS': re.compile(r'(truck|lorry|tipper|container|ಲಾರಿ)', re.IGNORECASE),
            'LIGHT_VEHICLE': re.compile(r'(auto|car|bike|lcv|ಆಟೋ)', re.IGNORECASE)
        }

        # ADDED: Extract structural civic agency entities responsible for clearance operations
        self.agency_patterns = {
            'BWSSB': re.compile(r'(water line|leakage|drainage|ಸೇವರೇಜ್|ಒಳಚರಂಡಿ|bwssb)', re.IGNORECASE),
            'BBMP': re.compile(r'(pothole|tree fall|asphalt|bbmp|ಗುಂಡಿ|ಮರ ಬಿದ್ದಿದೆ)', re.IGNORECASE),
            'BESCOM': re.compile(r'(wire|electric|pole|transformer|ಬೆಸ್ಕಾಂ)', re.IGNORECASE),
            'METRO_BMRCL': re.compile(r'(metro work|bmrcl|pillar|ಮೆಟ್ರೋ)', re.IGNORECASE)
        }

    def extract_features(self, text: str) -> dict:
        if not isinstance(text, str) or pd.isna(text):
            return {'extracted_severity': 'UNKNOWN', 'vehicle_class': 'UNKNOWN', 'action_agency': 'TRAFFIC_POLICE'}
        
        text_lower = text.lower()
        
        severity = 'NORMAL_OBSTRUCTION'
        for label, pattern in self.severity_patterns.items():
            if pattern.search(text_lower):
                severity = label
                break
                
        veh_class = 'OTHER'
        for label, pattern in self.vehicle_patterns.items():
            if pattern.search(text_lower):
                veh_class = label
                break

        # Extract responsible civic entity
        agency = 'TRAFFIC_POLICE'
        for label, pattern in self.agency_patterns.items():
            if pattern.search(text_lower):
                agency = label
                break
                
        return {
            'extracted_severity': severity,
            'vehicle_class': veh_class,
            'action_agency': agency
        }