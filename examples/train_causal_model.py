"""
Example script for training causal model.

This script demonstrates how to use CausalModelBuilder to train
Structural Causal Models (SCM) from causal graphs and data.
"""

import sys
import pandas as pd
import networkx as nx
from pathlib import Path
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal.causal_model import CausalModelBuilder
from src.causal.data_preprocessor import CausalDataPreprocessor

def main():
    """Main function."""
    # Configuration
    dataset_path = "datasets/OpenRCA/Bank"
    algorithm = 'varlingam'  # Algorithm used for graph construction
    
    # Input files
    data_file = "output/causal_data/all_data.csv"
    graph_file = f"output/causal_graph/{algorithm}/causal_graph.graphml"
    
    # Output
    output_dir = f"output/causal_model/{algorithm}"
    model_file = f"{output_dir}/scm_model.pkl"
    scm_graph_file = f"{output_dir}/scm_graph.graphml"
    
    # Model parameters
    causal_mechanism = 'auto'  # 'auto', 'linear', or 'nonlinear'
    
    print("=" * 60)
    print("Causal Model Training")
    print("=" * 60)
    print(f"Data file: {data_file}")
    print(f"Graph file: {graph_file}")
    print(f"Algorithm: {algorithm}")
    print(f"Causal mechanism: {causal_mechanism}")
    print(f"Output file: {model_file}")
    print()
    
    # Check if input files exist
    if not Path(data_file).exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please run examples/prepare_causal_data.py first.")
        return
        
    if not Path(graph_file).exists():
        print(f"Error: Graph file not found: {graph_file}")
        print(f"Please run examples/build_causal_graph.py (with algorithm='{algorithm}') first.")
        return
    
    # Load data
    print("Loading training data...")
    # Read CSV with datetime index
    # Note: CausalDataPreprocessor saves with 'timestamp_utc' or 'datetime' index
    try:
        training_data = pd.read_csv(data_file, index_col=0)
        # Convert index to datetime if it's not already (though for causal model it might not strictly matter if we drop it, 
        # but GCM might use it if it's time series specific. 
        # Actually standard GCM assumes IID samples, but we might have time series handling in future)
        # For now, just ensure it's loaded correctly.
        print(f"Loaded training data: {training_data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Load causal graph
    print("Loading causal graph...")
    try:
        causal_graph = nx.read_graphml(graph_file)
        print(f"Loaded causal graph: {causal_graph.number_of_nodes()} nodes, {causal_graph.number_of_edges()} edges")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return
        
    # Initialize model builder
    builder = CausalModelBuilder(verbose=True)
    
    # Build and train SCM
    print()
    print("=" * 60)
    print("Building and Training SCM")
    print("=" * 60)
    
    try:
        scm = builder.build_scm(
            causal_graph=causal_graph,
            training_data=training_data,
            causal_mechanism=causal_mechanism
        )
        
        # Save model
        print()
        print(f"Saving model to {model_file}...")
        builder.save_model(scm, model_file)

        # Save SCM graph with sanitized attributes
        try:
            original = scm.graph
            sanitized = nx.DiGraph()
            for n in original.nodes():
                attrs = {}
                try:
                    mech = scm.causal_mechanism(n)
                    attrs['mechanism'] = type(mech).__name__
                except Exception:
                    attrs['mechanism'] = 'UnknownMechanism'
                preds = list(original.predecessors(n)) if hasattr(original, 'predecessors') else []
                attrs['is_root'] = bool(len(preds) == 0)
                sanitized.add_node(n, **attrs)
            for u, v, data in original.edges(data=True):
                eattrs = {}
                for key in ('lag','p_value','coefficient','causal_strength','max_abs_coefficient','prior'):
                    if key in data:
                        val = data[key]
                        if isinstance(val, (int, float, bool, str)):
                            eattrs[key] = val
                        else:
                            try:
                                eattrs[key] = float(val)
                            except Exception:
                                eattrs[key] = str(val)
                sanitized.add_edge(u, v, **eattrs)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            nx.write_graphml(sanitized, scm_graph_file)
            print(f"Saved SCM graph to {scm_graph_file}")
        except Exception as e:
            print(f"Warning: Could not save SCM graph: {e}")
        
        print("Done!")
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
