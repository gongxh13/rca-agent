"""
Causal Graph Discovery Module

This module provides functionality for building causal graphs using causal-learn library.
It implements PC algorithm and other causal discovery methods.
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import json

try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    print("Warning: causal-learn not installed. Please install it: pip install causallearn")


class CausalGraphBuilder:
    """
    Builder for causal graphs using causal discovery algorithms.
    
    Uses PC algorithm from causal-learn to discover causal relationships
    from observational data.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        use_trace_prior: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the causal graph builder.
        
        Args:
            alpha: Significance level for independence tests (default: 0.05)
            use_trace_prior: Whether to use trace data as prior knowledge
            verbose: Whether to print progress information
        """
        if not CAUSAL_LEARN_AVAILABLE:
            raise ImportError(
                "causal-learn is required. Install it with: pip install causallearn"
            )
        
        self.alpha = alpha
        self.use_trace_prior = use_trace_prior
        self.verbose = verbose
        
    def build_causal_graph(
        self,
        data: pd.DataFrame,
        service_topology: Optional[nx.DiGraph] = None,
        background_knowledge: Optional[Any] = None
    ) -> nx.DiGraph:
        """
        Build causal graph from data using PC algorithm.
        
        Args:
            data: Wide table DataFrame (each row is a time point, each column is a metric)
            service_topology: Optional service topology graph from traces
            background_knowledge: Optional background knowledge for PC algorithm
            
        Returns:
            NetworkX DiGraph representing the causal graph
        """
        if data.empty:
            raise ValueError("Data is empty")
        
        if self.verbose:
            print(f"Building causal graph from {data.shape[0]} time points and {data.shape[1]} features...")
        
        # Prepare data for PC algorithm
        # Remove any remaining NaN values
        data_clean = data.dropna(axis=1, how='all')  # Remove columns with all NaN
        data_clean = data_clean.ffill().bfill()
        
        if data_clean.empty:
            raise ValueError("Data is empty after cleaning")
        
        # Remove constant columns (zero variance) - these cause singular matrix errors
        if self.verbose:
            print("Checking for constant columns...")
        constant_cols = []
        for col in data_clean.columns:
            if data_clean[col].nunique() <= 1 or data_clean[col].std() == 0:
                constant_cols.append(col)
        
        if constant_cols:
            if self.verbose:
                print(f"Removing {len(constant_cols)} constant columns: {constant_cols[:5]}...")
            data_clean = data_clean.drop(columns=constant_cols)
        
        if data_clean.empty:
            raise ValueError("Data is empty after removing constant columns")
        
        # Remove perfectly correlated columns (causes singular matrix)
        if self.verbose:
            print("Checking for perfectly correlated columns...")
        try:
            corr_matrix = data_clean.corr().abs()
            # Find pairs of columns with correlation = 1.0 (perfectly correlated)
            to_remove = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if pd.notna(corr_val) and corr_val >= 0.999:  # Near-perfect correlation
                        # Keep the first one, remove the second
                        col_i = corr_matrix.columns[i]
                        col_j = corr_matrix.columns[j]
                        if col_j not in to_remove:
                            to_remove.add(col_j)
                            if self.verbose:
                                print(f"  Removing {col_j} (perfectly correlated with {col_i}, r={corr_val:.4f})")
            
            if to_remove:
                data_clean = data_clean.drop(columns=list(to_remove))
                if self.verbose:
                    print(f"Removed {len(to_remove)} perfectly correlated columns")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not check for correlated columns: {e}")
            # Continue anyway
        
        if data_clean.empty:
            raise ValueError("Data is empty after removing correlated columns")
        
        # Check for remaining issues that could cause singular matrix
        # Check if any column is a linear combination of others
        if self.verbose:
            print("Final data check...")
        
        # Convert to numpy array
        data_array = data_clean.values
        
        # Check for NaN or Inf values
        if np.isnan(data_array).any() or np.isinf(data_array).any():
            if self.verbose:
                print("Warning: Data contains NaN or Inf values, replacing...")
            data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Final check: ensure matrix is not singular
        # Note: We can't directly fix the correlation matrix used by PC algorithm,
        # but we've removed problematic columns above
        # The PC algorithm will handle remaining issues internally
        if self.verbose:
            # Check condition number as a warning
            try:
                corr_check = np.corrcoef(data_array.T)
                cond_num = np.linalg.cond(corr_check)
                if cond_num > 1e10:
                    print(f"Warning: Correlation matrix condition number is high ({cond_num:.2e}), "
                          f"may cause numerical issues")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not check correlation matrix condition: {e}")
        
        if self.verbose:
            print(f"Using {data_array.shape[0]} samples and {data_array.shape[1]} variables")
        
        # Build background knowledge if trace topology is provided
        background_knowledge_matrix = None
        if self.use_trace_prior and service_topology is not None:
            background_knowledge_matrix = self._build_background_knowledge(
                data_clean.columns.tolist(),
                service_topology
            )
        
        # Run PC algorithm
        if self.verbose:
            print("Running PC algorithm...")
            if background_knowledge_matrix is not None:
                print("Using trace topology as background knowledge")
        
        cg = pc(
            data_array,
            alpha=self.alpha,
            indep_test=fisherz,
            background_knowledge=background_knowledge_matrix
        )
        
        # Convert to NetworkX graph
        causal_graph = self._convert_to_networkx(cg, data_clean.columns.tolist())
        
        if self.verbose:
            print(f"Causal graph: {causal_graph.number_of_nodes()} nodes, "
                  f"{causal_graph.number_of_edges()} edges")
        
        return causal_graph
    
    def _build_background_knowledge(
        self,
        feature_names: List[str],
        service_topology: nx.DiGraph
    ) -> Optional[np.ndarray]:
        """
        Build background knowledge matrix from service topology.
        
        Currently returns None to let the algorithm discover causal relationships
        purely from data without topology constraints.
        
        Args:
            feature_names: List of feature names (column names from wide table)
            service_topology: Service topology graph from traces (currently not used)
            
        Returns:
            None (no background knowledge constraints)
        """
        # Return None to indicate no constraints
        # This allows the PC algorithm to discover causal relationships purely from data
        if self.verbose:
            print("Not using topology constraints - letting algorithm discover from data")
        
        return None
    
    def _convert_to_networkx(
        self,
        cg: Any,
        feature_names: List[str]
    ) -> nx.DiGraph:
        """
        Convert causal-learn graph to NetworkX DiGraph.
        
        Args:
            cg: Causal graph from PC algorithm
            feature_names: List of feature names
            
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # Add nodes
        for i, name in enumerate(feature_names):
            G.add_node(name)
        
        # Add edges from adjacency matrix
        adj_matrix = cg.G.graph
        
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                if adj_matrix[i, j] != 0:
                    # Check if edge is directed
                    if adj_matrix[j, i] == 0:
                        # Directed edge: i -> j
                        G.add_edge(feature_names[i], feature_names[j])
                    else:
                        # Undirected edge: add as directed (PC algorithm may not determine direction)
                        # We'll add it as i -> j for now
                        if feature_names[i] < feature_names[j]:  # Avoid duplicate
                            G.add_edge(feature_names[i], feature_names[j])
        
        return G
    
    def optimize_graph(
        self,
        graph: nx.DiGraph,
        service_topology: Optional[nx.DiGraph] = None
    ) -> nx.DiGraph:
        """
        Optimize causal graph by removing unreasonable edges.
        
        Args:
            graph: Causal graph to optimize
            service_topology: Optional service topology for validation
            
        Returns:
            Optimized causal graph
        """
        if self.verbose:
            print("Optimizing causal graph...")
        
        G = graph.copy()
        edges_to_remove = []
        
        # Remove self-loops
        for node in list(G.nodes()):
            if G.has_edge(node, node):
                edges_to_remove.append((node, node))
        
        # Remove edges that violate service topology constraints
        if service_topology is not None:
            # If service A is downstream of service B in topology,
            # B's metrics should not affect A's metrics (upstream affects downstream)
            # Actually, this is the opposite - upstream should affect downstream
            # So we keep edges that follow topology direction
            pass  # Can add more sophisticated validation
        
        # Remove edges
        for edge in edges_to_remove:
            if G.has_edge(*edge):
                G.remove_edge(*edge)
        
        if self.verbose:
            print(f"After optimization: {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges")
        
        return G
    
    def save_graph(
        self,
        graph: nx.DiGraph,
        output_path: str,
        format: str = 'graphml'
    ):
        """
        Save causal graph to file.
        
        Args:
            graph: Causal graph to save
            output_path: Output file path
            format: File format ('graphml', 'gml', 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'graphml':
            nx.write_graphml(graph, output_path)
        elif format == 'gml':
            nx.write_gml(graph, output_path)
        elif format == 'json':
            graph_data = {
                'nodes': list(graph.nodes()),
                'edges': list(graph.edges())
            }
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if self.verbose:
            print(f"Saved causal graph to {output_path}")
    
    def save_edges_csv(
        self,
        graph: nx.DiGraph,
        output_path: str
    ):
        """
        Save causal graph edges to CSV file.
        
        Args:
            graph: Causal graph
            output_path: Output CSV file path
        """
        edges = list(graph.edges())
        edges_df = pd.DataFrame(edges, columns=['source', 'target'])
        edges_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"Saved {len(edges)} edges to {output_path}")
    
    def load_graph(
        self,
        input_path: str,
        format: str = 'graphml'
    ) -> nx.DiGraph:
        """
        Load causal graph from file.
        
        Args:
            input_path: Input file path
            format: File format ('graphml', 'gml', 'json')
            
        Returns:
            NetworkX DiGraph
        """
        input_path = Path(input_path)
        
        if format == 'graphml':
            graph = nx.read_graphml(input_path)
        elif format == 'gml':
            graph = nx.read_gml(input_path)
        elif format == 'json':
            with open(input_path, 'r') as f:
                graph_data = json.load(f)
            graph = nx.DiGraph()
            graph.add_nodes_from(graph_data['nodes'])
            graph.add_edges_from(graph_data['edges'])
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if self.verbose:
            print(f"Loaded causal graph: {graph.number_of_nodes()} nodes, "
                  f"{graph.number_of_edges()} edges")
        
        return graph
    
    def get_graph_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Get statistics about the causal graph.
        
        Args:
            graph: Causal graph
            
        Returns:
            Dictionary with graph statistics
        """
        return {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_dag': nx.is_directed_acyclic_graph(graph),
            'num_components': nx.number_weakly_connected_components(graph),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        }

