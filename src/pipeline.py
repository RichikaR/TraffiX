import pandas as pd
import numpy as np
from nlp_engine import TrafficNLPEngine
from graph_engine import BengaluruSpatialGraph

def process_pipeline(input_csv_path: str, output_csv_path: str):
    print("Initializing components...")
    nlp = TrafficNLPEngine()
    spatial_graph = BengaluruSpatialGraph()

    print("Loading raw Astram event data...")
    # Explicitly parsing string datetimes to handle time calculation accurately
    df = pd.read_csv(input_csv_path, low_memory=False)
    
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce')

    print("Engineering target metrics (Historical Clearance Duration)...")
    # Drop rows where execution times are corrupted or missing to prevent training noise
    df = df.dropna(subset=['start_datetime', 'closed_datetime'])
    
    # Calculate target time in minutes
    df['target_clearance_duration_mins'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
    # Clean anomalies (e.g., negative duration or events lasting longer than 24 hours)
    df = df[(df['target_clearance_duration_mins'] > 0) & (df['target_clearance_duration_mins'] < 1440)]

    print("Processing NLP features from commentary columns...")
    nlp_features = df['comment'].fillna(df['description']).apply(nlp.extract_features)
    nlp_df = pd.DataFrame(nlp_features.tolist(), index=df.index)
    df = pd.concat([df, nlp_df], axis=1)

    print("Processing Spatial Network Snapping (Vectorized)...")
    spatial_features = spatial_graph.snap_coordinates_to_network_vectorized(
        df['latitude'].values, df['longitude'].values
    )
    spatial_df = pd.DataFrame(spatial_features, index=df.index)
    final_dataset = pd.concat([df, spatial_df], axis=1)

    # Filter out clean columns to form our definitive ML Feature Store matrix
    feature_columns = [
        'id', 'event_type', 'event_cause', 'priority', 'corridor', 'zone',
        'target_clearance_duration_mins', 'extracted_severity', 'vehicle_class',
        'action_agency',   # Bug fix: was silently dropped from output in v1
        'osm_edge_u', 'osm_edge_v', 'road_highway_type', 'road_lanes', 'road_maxspeed'
    ]
    
    output_df = final_dataset[feature_columns]
    output_df.to_csv(output_csv_path, index=False)
    print(f"Pipeline complete! Engineered feature set saved to: {output_csv_path}")

if __name__ == "__main__":
    import os, glob
    # Find the CSV however it was named (spaces or underscores)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    candidates = glob.glob(os.path.join(data_dir, '*Astram*anonymized*.csv'))
    if not candidates:
        raise FileNotFoundError(
            f"No ASTRAM CSV found in {data_dir}. "
            "Copy the dataset CSV into the data/ folder and re-run."
        )
    input_csv = candidates[0]
    print(f"Using dataset: {input_csv}")
    process_pipeline(
        input_csv_path=input_csv,
        output_csv_path=os.path.join(data_dir, "engineered_traffic_features.csv")
    )