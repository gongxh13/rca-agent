"""
Test script to verify data loading works correctly.

This script tests that CausalDataPreprocessor can correctly load
data from CSV files with different timestamp formats.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal.data_preprocessor import CausalDataPreprocessor
import pandas as pd


def test_data_loading():
    """Test data loading functionality."""
    print("=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = CausalDataPreprocessor(
        dataset_path="datasets/OpenRCA/Bank",
        time_granularity="5min"
    )
    
    # Test loading container metrics
    print("\n1. Testing container metrics loading...")
    container_df = preprocessor.load_all_data(
        start_date="2021-03-04",
        end_date="2021-03-04",
        metric_type="container"
    )
    
    if not container_df.empty:
        print(f"   ✓ Loaded {len(container_df)} container metric records")
        print(f"   ✓ Date range: {container_df['datetime'].min()} to {container_df['datetime'].max()}")
        print(f"   ✓ Timezone: {container_df['datetime'].iloc[0].tz}")
        print(f"   ✓ Sample columns: {list(container_df.columns)}")
    else:
        print("   ✗ No container metrics loaded")
    
    # Test loading app metrics
    print("\n2. Testing app metrics loading...")
    app_df = preprocessor.load_all_data(
        start_date="2021-03-04",
        end_date="2021-03-04",
        metric_type="app"
    )
    
    if not app_df.empty:
        print(f"   ✓ Loaded {len(app_df)} app metric records")
        print(f"   ✓ Date range: {app_df['datetime'].min()} to {app_df['datetime'].max()}")
        print(f"   ✓ Timezone: {app_df['datetime'].iloc[0].tz}")
        print(f"   ✓ Sample columns: {list(app_df.columns)}")
    else:
        print("   ✗ No app metrics loaded")
    
    # Test loading trace data
    print("\n3. Testing trace data loading...")
    trace_df = preprocessor._load_trace_file("2021_03_04")
    
    if trace_df is not None and not trace_df.empty:
        print(f"   ✓ Loaded {len(trace_df)} trace records")
        print(f"   ✓ Date range: {trace_df['datetime'].min()} to {trace_df['datetime'].max()}")
        print(f"   ✓ Timezone: {trace_df['datetime'].iloc[0].tz}")
        print(f"   ✓ Sample columns: {list(trace_df.columns)}")
    else:
        print("   ✗ No trace data loaded")
    
    # Test extracting core metrics
    print("\n4. Testing core metrics extraction...")
    if not container_df.empty:
        core_metrics = preprocessor.extract_core_metrics(container_df)
        print(f"   ✓ Extracted {len(core_metrics)} core metric records")
        print(f"   ✓ Unique components: {core_metrics['cmdb_id'].nunique()}")
        print(f"   ✓ Unique metrics: {core_metrics['kpi_name'].nunique()}")
        print(f"   ✓ Sample metrics: {core_metrics['kpi_name'].unique()[:5].tolist()}")
    
    # Test building service topology
    print("\n5. Testing service topology construction...")
    topology = preprocessor.build_service_topology("2021-03-04", "2021-03-04")
    
    if topology.number_of_nodes() > 0:
        print(f"   ✓ Topology has {topology.number_of_nodes()} nodes")
        print(f"   ✓ Topology has {topology.number_of_edges()} edges")
        print(f"   ✓ Sample nodes: {list(topology.nodes())[:5]}")
    else:
        print("   ✗ Empty topology")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_loading()

