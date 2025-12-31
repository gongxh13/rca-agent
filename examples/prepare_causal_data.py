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
    end_date = "2021-03-05"  # Use multiple days for better causal discovery
    time_granularity = "10min"  # 5 minutes for causal graph construction
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
    # compact_mode=True (default): Only keep key metrics based on fault types in record.csv
    # compact_mode=False: Keep all core metrics (original behavior)
    # service_aggregation=True: Aggregate metrics for same-type services (e.g. Tomcat01, Tomcat02 -> Tomcat)
    #                          to avoid multicollinearity and reduce graph size
    preprocessor = CausalDataPreprocessor(
        dataset_path=dataset_path,
        time_granularity=time_granularity,
        compact_mode=True,  # Enable compact mode (default: True)
        service_aggregation=True  # Enable service aggregation
    )
    
    # Define selected business services (optional)
    # If provided, only these services will be included in app metrics
    # Example: ['servicetest1', 'servicetest2', ...]
    # Leave as None to include all services
    selected_business_services = None
    # selected_business_services = [f'servicetest{i}' for i in range(1, 12)] # Example: Select specific services
    
    # Prepare data with memory optimization
    # chunksize: Read files in chunks to save memory (100000 rows per chunk)
    # trace_sample_ratio: Sample 10% of trace data for topology building (sufficient for dependency discovery)
    results = preprocessor.prepare_causal_data(
        start_date=start_date,
        end_date=end_date,
        include_app_metrics=include_app_metrics,
        chunksize=100000,  # Read files in chunks to save memory
        trace_sample_ratio=0.1,  # Sample 10% of trace data (sufficient for topology)
        selected_business_services=selected_business_services  # Filter business services
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

