"""
Example script for building causal graph.

This script demonstrates how to use CausalGraphBuilder to construct
causal graphs from preprocessed data.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal.data_preprocessor import CausalDataPreprocessor
from src.causal.causal_discovery import CausalGraphBuilder
import networkx as nx


def main():
    """Main function."""
    # Configuration
    dataset_path = "datasets/OpenRCA/Bank"
    data_file = "outputs/causal_data/all_data.csv"
    topology_file = "outputs/causal_data/service_topology.graphml"
    output_dir = "outputs/causal_graph"
    alpha = 0.05
    use_trace_prior = True
    algorithm = 'varlingam'  # Options: 'pc', 'pcmci', 'granger', 'varlingam', 'granger_pc'
    # 'granger' is recommended for fast analysis (good for large datasets)
    # 'varlingam' is recommended for balanced speed and accuracy
    # 'pcmci' is most accurate but slow
    max_lag = 5  # Maximum time lag (used for time series algorithms)
    varlingam_threshold = 0.01  # For VARLiNGAM: threshold for edge significance
    # Smaller values produce more edges. Recommended: 0.01 (strict), 0.001 (moderate), 1e-6 (very loose)
    
    print("=" * 60)
    print("Causal Graph Construction")
    print("=" * 60)
    print(f"Data file: {data_file}")
    print(f"Topology file: {topology_file}")
    print(f"Algorithm: {algorithm}")
    if algorithm in ['pcmci', 'granger', 'varlingam']:
        print(f"Max lag: {max_lag}")
    print(f"Alpha (significance level): {alpha}")
    print(f"Use trace prior: {use_trace_prior}")
    print()
    
    # Load data
    print("Loading data...")
    if Path(data_file).exists():
        # Load preprocessed data
        wide_table = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"Loaded wide table: {wide_table.shape}")
        
        # Load service topology
        service_topology = None
        if Path(topology_file).exists():
            service_topology = nx.read_graphml(topology_file)
            print(f"Loaded service topology: {service_topology.number_of_nodes()} nodes, "
                  f"{service_topology.number_of_edges()} edges")
    else:
        # Data file not found, run preprocessing
        print(f"Data file not found: {data_file}")
        print("Running data preprocessing first...")
        
        # Run preprocessing
        preprocessor = CausalDataPreprocessor(
            dataset_path=dataset_path,
            time_granularity="10min"
        )
        results = preprocessor.prepare_causal_data(
            start_date="2021-03-04",
            end_date="2021-03-05",
            include_app_metrics=True
        )
        wide_table = results['wide_table']
        service_topology = results['service_topology']
        
        # Save for next time
        Path("outputs/causal_data").mkdir(parents=True, exist_ok=True)
        preprocessor.save_results(results, "outputs/causal_data", datetime_as_timestamp=True)
    
    # Initialize causal graph builder
    builder = CausalGraphBuilder(
        alpha=alpha,
        use_trace_prior=use_trace_prior,
        verbose=True,
        algorithm=algorithm,
        max_lag=max_lag,
        varlingam_threshold=varlingam_threshold,  # Control edge density for VARLiNGAM
        remove_multicollinearity=True,
    )
    
    # Build causal graph
    print()
    print("=" * 60)
    print("Building causal graph...")
    print("=" * 60)
    causal_graph = builder.build_causal_graph(
        data=wide_table,
        service_topology=service_topology
    )
    
    # Optimize graph
    print()
    print("Optimizing causal graph...")
    causal_graph = builder.optimize_graph(
        graph=causal_graph,
        service_topology=service_topology
    )
    
    # Get statistics
    stats = builder.get_graph_statistics(causal_graph)
    print()
    print("=" * 60)
    print("Graph Statistics")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save graph
    print()
    print("=" * 60)
    print("Saving causal graph...")
    print("=" * 60)
    # Create algorithm-specific output directory to avoid overwriting
    algorithm_output_dir = Path(output_dir) / algorithm
    algorithm_output_dir.mkdir(parents=True, exist_ok=True)
    
    builder.save_graph(
        graph=causal_graph,
        output_path=str(algorithm_output_dir / "causal_graph.graphml"),
        format='graphml'
    )
    
    builder.save_edges_csv(
        graph=causal_graph,
        output_path=str(algorithm_output_dir / "causal_edges.csv")
    )
    
    # Save statistics
    import json
    with open(algorithm_output_dir / "graph_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print()
    print("Done!")
    print(f"Causal graph saved to {algorithm_output_dir}/")


if __name__ == "__main__":
    main()
