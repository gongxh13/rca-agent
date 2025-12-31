"""
Example script for running root cause analysis.

This script demonstrates how to use RootCauseAnalyzer to identify
root causes of anomalies using trained SCM models.
"""

import sys
import pandas as pd
import networkx as nx
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal.root_cause_inference import RootCauseAnalyzer
from src.causal.causal_model import CausalModelBuilder
from src.causal.data_preprocessor import CausalDataPreprocessor

def main():
    """Main function."""
    # Configuration
    dataset_path = "datasets/OpenRCA/Bank"
    algorithm = 'varlingam'
    
    # Input files
    model_file = f"output/causal_model/{algorithm}/scm_model.pkl"
    
    # Fault scenario configuration
    # Example fault: ServiceTest1 latency increase
    # In a real scenario, these would come from an anomaly detector or user query
    fault_start_time = "2021-03-04 22:00:00"
    fault_end_time = "2021-03-04 22:30:00"
    FAULT_TIMEZONE = "Asia/Shanghai"  # Use "UTC" if timestamps are UTC; use "Asia/Shanghai" for 东8区
    target_node = None  # If None, auto-detect anomalous target(s)
    
    # Note: If target node doesn't exist in your graph, change it to a valid node
    # You can check valid nodes in the loaded graph/model
    
    print("=" * 60)
    print("Root Cause Analysis")
    print("=" * 60)
    print(f"Model file: {model_file}")
    print(f"Fault time range: {fault_start_time} to {fault_end_time} ({FAULT_TIMEZONE})")
    print(f"Target anomaly: {target_node}")
    print()
    
    # Check if model exists
    if not Path(model_file).exists():
        print(f"Error: Model file not found: {model_file}")
        print("Please run examples/train_causal_model.py first.")
        return
        
    # Load model
    print("Loading causal model...")
    builder = CausalModelBuilder(verbose=False)
    try:
        scm = builder.load_model(model_file)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # If target_node is provided, validate it; else we will auto-detect later
    if target_node is not None and target_node not in scm.graph.nodes:
        print(f"Warning: Target node '{target_node}' not found in model.")
        print("Available nodes (first 10):", list(scm.graph.nodes)[:10])
        target_node = None

    # Load anomaly data
    # We need to load data for the specific fault time window
    print("Loading anomaly data...")
    preprocessor = CausalDataPreprocessor(
        dataset_path=dataset_path,
        time_granularity="1min", # Use finer granularity for RCA
        compact_mode=True,
        service_aggregation=True  # Enable service aggregation to match training data
    )
    
    # Define selected business services (should match training configuration)
    # selected_business_services = None
    selected_business_services = None# Example
    
    # Note: We need to ensure we get the same features as the training data
    # Ideally we should load from the same processed data or process it identically
    # Here we reload from raw to demonstrate handling a specific window
    try:
        results = preprocessor.prepare_causal_data(
            start_date=fault_start_time.split()[0], # Extract date part
            end_date=fault_end_time.split()[0],
            include_app_metrics=True,
            chunksize=100000,
            selected_business_services=selected_business_services
        )
        wide_table = results['wide_table']
        
        # Filter for the specific time window
        # Ensure index is datetime
        if not isinstance(wide_table.index, pd.DatetimeIndex):
             # Try to convert if it's not already (it might be timestamp)
             # preprocessor saves as timestamp or string, but here it returns dataframe directly
             # create_wide_table returns df with datetime index usually
             pass
             
        # Filter by time range
        # Note: wide_table index is usually datetime objects if coming directly from preprocessor
        # If loaded from CSV, needs parsing. Here we got it from preprocessor directly.
        
        # However, preprocessor.prepare_causal_data returns the whole day(s) data
        # We need to filter for the specific window
        
        # Handle index type (might be int timestamp if saved/loaded, but here it's fresh df)
        # In create_wide_table, it returns df with datetime index.
        
        # Convert string times to timezone-aware datetime
        start_dt_local = pd.to_datetime(fault_start_time)
        end_dt_local = pd.to_datetime(fault_end_time)
        if FAULT_TIMEZONE and FAULT_TIMEZONE.upper() != "UTC":
            try:
                start_dt_local = start_dt_local.tz_localize(FAULT_TIMEZONE)
                end_dt_local = end_dt_local.tz_localize(FAULT_TIMEZONE)
                start_dt = start_dt_local.tz_convert('UTC')
                end_dt = end_dt_local.tz_convert('UTC')
            except Exception:
                # Fallback: treat as local naive then convert by offset (Asia/Shanghai = +8)
                start_dt = start_dt_local.tz_localize('UTC')
                end_dt = end_dt_local.tz_localize('UTC')
        else:
            start_dt = start_dt_local.tz_localize('UTC')
            end_dt = end_dt_local.tz_localize('UTC')
        
        # Ensure index is timezone aware (UTC) to match
        if wide_table.index.tz is None:
            wide_table.index = wide_table.index.tz_localize('UTC')
            
        anomaly_data = wide_table[
            (wide_table.index >= start_dt) & 
            (wide_table.index <= end_dt)
        ]
        
        print(f"Loaded {len(anomaly_data)} anomaly samples.")
        
        if len(anomaly_data) == 0:
            print("Error: No data found for the specified time range.")
            return
            
        # Ensure columns match model
        # The model might expect specific columns. 
        # GCM handles missing columns by ignoring them or erroring?
        # Ideally anomaly_data should have same columns as training data.
        # If training data had columns dropped (e.g. constant), we need to handle that.
        # For this example, we assume consistency.
        
    except Exception as e:
        print(f"Error preparing anomaly data: {e}")
        return

    # Run RCA
    print()
    print("=" * 60)
    print("Running Root Cause Analysis")
    print("=" * 60)
    
    analyzer = RootCauseAnalyzer(verbose=True)

    try:
        if target_node is None:
            # Auto-detect anomalous targets by z-score within the fault window
            # Prefer service metrics (mrt, sr)
            means = wide_table.mean()
            stds = wide_table.std().replace(0, 1.0)
            # Use average over window to compute deviation
            window_avg = anomaly_data.mean()
            zscores = ((window_avg - means) / stds).abs().sort_values(ascending=False)
            candidate_targets = [c for c in zscores.index if c.endswith('_mrt') or c.endswith('_sr')]
            if not candidate_targets:
                candidate_targets = list(zscores.index)[:5]
            else:
                candidate_targets = candidate_targets[:5]
            print("Auto-selected target candidates:", candidate_targets)
            # Analyze top-1 and print batch results for top-5
            result = analyzer.analyze(
                model=scm,
                target_node=candidate_targets[0],
                anomaly_data=anomaly_data,
                n_top=5
            )
        else:
            result = analyzer.analyze(
                model=scm,
                target_node=target_node,
                anomaly_data=anomaly_data,
                n_top=5
            )
        
        print()
        print("Top Root Causes:")
        print("-" * 30)
        for i, cause in enumerate(result['root_causes']):
            print(f"{i+1}. {cause['node']}")
            print(f"   Score: {cause['score']:.4f}")
            print(f"   Contribution: {cause['contribution']:.4f}")
            print(f"   Component: {cause['component']}")
            print(f"   Metric: {cause['metric']}")
            print()
            
        # Save results
        out_target = result['target_node']
        output_file = f"output/rca_result_{out_target}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error during RCA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
