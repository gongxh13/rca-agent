"""
Causal Model Module

This module provides functionality for building and training Structural Causal Models (SCM)
using DoWhy-GCM based on discovered causal graphs.
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import pickle
import json

try:
    import dowhy.gcm as gcm
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False


class CausalModelBuilder:
    """
    Builder for Structural Causal Models (SCM) using DoWhy-GCM.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the causal model builder.
        
        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        
        if not DOWHY_AVAILABLE:
            raise ImportError(
                "DoWhy library is not available. "
                "Please install it with: pip install dowhy"
            )
            
    def build_scm(
        self,
        causal_graph: nx.DiGraph,
        training_data: pd.DataFrame,
        causal_mechanism: str = 'auto'
    ) -> Any:
        """
        Build and train a Structural Causal Model (SCM) from a causal graph.
        
        Args:
            causal_graph: NetworkX DiGraph representing the causal structure
            training_data: DataFrame containing training data (columns must match graph nodes)
            causal_mechanism: Type of causal mechanism to assign ('auto', 'linear', 'nonlinear')
            
        Returns:
            Trained GCM model
        """
        if self.verbose:
            print(f"Building SCM from graph with {causal_graph.number_of_nodes()} nodes and {causal_graph.number_of_edges()} edges...")
            
        # Validate data columns match graph nodes
        graph_nodes = set(causal_graph.nodes())
        data_columns = set(training_data.columns)
        
        missing_cols = graph_nodes - data_columns
        if missing_cols:
            raise ValueError(f"Training data is missing columns for nodes: {missing_cols}")
            
        # Ensure DAG
        if not nx.is_directed_acyclic_graph(causal_graph):
            if self.verbose:
                print("Input graph contains cycles; converting to DAG...")
            causal_graph = self._ensure_dag(causal_graph)
            if self.verbose:
                print(f"DAG check: {nx.is_directed_acyclic_graph(causal_graph)}")
        
        # Create GCM model from NetworkX graph
        scm = gcm.InvertibleStructuralCausalModel(causal_graph)
        
        # Assign causal mechanisms
        if self.verbose:
            print(f"Assigning causal mechanisms ({causal_mechanism})...")
            
        if causal_mechanism == 'auto':
            gcm.auto.assign_causal_mechanisms(scm, training_data)
        elif causal_mechanism == 'linear':
            for node in scm.graph.nodes:
                if list(scm.graph.predecessors(node)):
                    scm.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
                else:
                    scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        elif causal_mechanism == 'nonlinear':
             for node in scm.graph.nodes:
                if list(scm.graph.predecessors(node)):
                    scm.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_nonlinear_regressor()))
                else:
                    scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        else:
             raise ValueError(f"Unknown causal mechanism: {causal_mechanism}")

        # Fit the model
        if self.verbose:
            print("Fitting SCM model...")
            
        gcm.fit(scm, training_data)
        
        if self.verbose:
            print("SCM model training completed.")
            
        return scm
    
    def _ensure_dag(self, graph: nx.DiGraph) -> nx.DiGraph:
        G = graph.copy()
        max_iter = max(1, G.number_of_edges())
        for _ in range(max_iter):
            if nx.is_directed_acyclic_graph(G):
                break
            cycles = list(nx.simple_cycles(G))
            if not cycles:
                break
            cycle = cycles[0]
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
            best_edge = None
            best_score = -1e9
            for u, v in edges:
                data = G.get_edge_data(u, v) or {}
                prior_flag = 1 if data.get('prior', False) else 0
                p_val = float(data.get('p_value', 1.0))
                strength = data.get('causal_strength', data.get('max_abs_coefficient', data.get('coefficient', 0.0)))
                try:
                    strength = float(strength)
                except Exception:
                    strength = 0.0
                score = prior_flag * 10 + p_val - abs(strength)
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            if best_edge is not None:
                if self.verbose:
                    print(f"Removing edge {best_edge} to break cycle")
                G.remove_edge(*best_edge)
            else:
                break
        return G
    
    def evaluate_model(
        self,
        model: Any,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate the trained SCM model.
        
        Args:
            model: Trained GCM model
            test_data: DataFrame containing test data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.verbose:
            print("Evaluating SCM model...")
            
        # Calculate evaluation metrics (e.g., NRMSE for each node)
        # This is a simplified evaluation. DoWhy-GCM provides more advanced evaluation methods.
        metrics = {}
        
        # TODO: Implement more comprehensive evaluation if needed
        # For now, we rely on the fitting process
        
        return metrics

    def save_model(self, model: Any, output_path: str):
        """
        Save the trained SCM model to a file.
        
        Args:
            model: Trained GCM model
            output_path: Path to save the model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)
            if self.verbose:
                print(f"Saved SCM model to {output_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            
    def load_model(self, input_path: str) -> Any:
        """
        Load a trained SCM model from a file.
        
        Args:
            input_path: Path to the model file
            
        Returns:
            Trained GCM model
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Model file not found: {input_path}")
            
        try:
            with open(input_path, 'rb') as f:
                model = pickle.load(f)
            if self.verbose:
                print(f"Loaded SCM model from {input_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
