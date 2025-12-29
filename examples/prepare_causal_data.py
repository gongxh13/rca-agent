"""
Example script for preparing causal analysis data.

This script demonstrates how to use CausalDataPreprocessor to prepare
OpenRCA data for causal graph construction.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal.data_preprocessor import CausalDataPreprocessor


def main():
    """Main function."""
    # Configuration
    dataset_path = "datasets/OpenRCA/Bank"
    start_date = "2021-03-04"
    end_date = "2021-03-25"  # Use multiple days for better causal discovery
    time_granularity = "5min"  # 5 minutes for causal graph construction
    output_dir = "output/causal_data"
    include_app_metrics = True
    
    print("=" * 60)
    print("Causal Data Preprocessing")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Time granularity: {time_granularity}")
    print(f"Include app metrics: {include_app_metrics}")
    print()
    
    # Initialize preprocessor
    preprocessor = CausalDataPreprocessor(
        dataset_path=dataset_path,
        time_granularity=time_granularity
    )
    
    # Prepare data with memory optimization
    # chunksize: Read files in chunks to save memory (100000 rows per chunk)
    # trace_sample_ratio: Sample 10% of trace data for topology building (sufficient for dependency discovery)
    results = preprocessor.prepare_causal_data(
        start_date=start_date,
        end_date=end_date,
        include_app_metrics=include_app_metrics,
        chunksize=100000,  # Read files in chunks to save memory
        trace_sample_ratio=0.1  # Sample 10% of trace data (sufficient for topology)
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Wide table shape: {results['wide_table'].shape}")
    print(f"Service topology: {results['service_topology'].number_of_nodes()} nodes, "
          f"{results['service_topology'].number_of_edges()} edges")
    print()
    print("Data alignment report:")
    print(f"  Container components: {len(results['data_alignment_report']['container_components'])}")
    print(f"  App services: {len(results['data_alignment_report']['app_services'])}")
    print()
    print("Data quality report:")
    print(f"  Missing rate: {results['data_quality_report'].get('missing_rate', 0):.2%}")
    print(f"  Feature count: {results['data_quality_report'].get('feature_count', 0)}")
    print(f"  Time points: {results['data_quality_report'].get('time_points', 0)}")
    
    # Save results
    print()
    print("=" * 60)
    print("Saving results...")
    print("=" * 60)
    # Save datetime as UTC timestamp (more efficient, avoids timezone issues)
    # Set datetime_as_timestamp=False to save as ISO string instead
    preprocessor.save_results(results, output_dir, datetime_as_timestamp=True)
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()

