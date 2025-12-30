"""
Test script for compact mode in data preprocessing.

This script demonstrates the difference between compact mode and normal mode.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal.data_preprocessor import CausalDataPreprocessor
import pandas as pd


def compare_modes():
    """Compare compact mode vs normal mode."""
    dataset_path = "datasets/OpenRCA/Bank"
    start_date = "2021-03-04"
    end_date = "2021-03-04"  # Single day for quick test
    time_granularity = "5min"
    
    print("=" * 80)
    print("Comparing Compact Mode vs Normal Mode")
    print("=" * 80)
    print()
    
    # Test compact mode
    print("Testing COMPACT MODE (fault-based metric selection)...")
    print("-" * 80)
    preprocessor_compact = CausalDataPreprocessor(
        dataset_path=dataset_path,
        time_granularity=time_granularity,
        compact_mode=True
    )
    
    # Load a small sample to check metrics
    container_df_compact = preprocessor_compact.load_all_data(
        start_date, end_date, metric_type='container', chunksize=100000
    )
    print(f"Loaded {len(container_df_compact)} container metric records")
    
    # Extract core metrics in compact mode
    metrics_compact = preprocessor_compact.extract_core_metrics(container_df_compact)
    unique_metrics_compact = metrics_compact['kpi_name'].unique()
    
    print(f"\nCompact mode results:")
    print(f"  Total records: {len(metrics_compact)}")
    print(f"  Unique metrics: {len(unique_metrics_compact)}")
    print(f"  Metrics: {sorted(unique_metrics_compact)}")
    
    # Test normal mode
    print("\n" + "=" * 80)
    print("Testing NORMAL MODE (all core metrics)...")
    print("-" * 80)
    preprocessor_normal = CausalDataPreprocessor(
        dataset_path=dataset_path,
        time_granularity=time_granularity,
        compact_mode=False
    )
    
    # Load same data
    container_df_normal = preprocessor_normal.load_all_data(
        start_date, end_date, metric_type='container', chunksize=100000
    )
    print(f"Loaded {len(container_df_normal)} container metric records")
    
    # Extract core metrics in normal mode
    metrics_normal = preprocessor_normal.extract_core_metrics(container_df_normal)
    unique_metrics_normal = metrics_normal['kpi_name'].unique()
    
    print(f"\nNormal mode results:")
    print(f"  Total records: {len(metrics_normal)}")
    print(f"  Unique metrics: {len(unique_metrics_normal)}")
    print(f"  Metrics: {sorted(unique_metrics_normal)}")
    
    # Compare
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"Compact mode metrics: {len(unique_metrics_compact)}")
    print(f"Normal mode metrics: {len(unique_metrics_normal)}")
    print(f"Reduction: {len(unique_metrics_normal) - len(unique_metrics_compact)} metrics removed")
    print(f"Reduction rate: {(1 - len(unique_metrics_compact) / len(unique_metrics_normal)) * 100:.1f}%")
    
    # Show removed metrics
    removed_metrics = set(unique_metrics_normal) - set(unique_metrics_compact)
    if removed_metrics:
        print(f"\nRemoved metrics in compact mode:")
        for metric in sorted(removed_metrics):
            print(f"  - {metric}")
    
    # Show fault types detected
    if preprocessor_compact.fault_types:
        print(f"\nFault types detected from record.csv:")
        for fault_type in sorted(preprocessor_compact.fault_types.keys()):
            print(f"  - {fault_type}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    compare_modes()

