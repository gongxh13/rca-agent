"""
Causal Graph Discovery Module

This module provides functionality for building causal graphs using causal discovery algorithms.
Supports both PC algorithm (causal-learn) and PCMCI+ algorithm (tigramite).
PCMCI+ is recommended for time series data as it handles temporal dependencies better.
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import json

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

from tigramite import data_processing as dp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


class CausalGraphBuilder:
    """
    Builder for causal graphs using causal discovery algorithms.
    
    Supports both PC algorithm (causal-learn) and PCMCI+ algorithm (tigramite).
    PCMCI+ is recommended for time series data as it handles temporal dependencies better.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        use_trace_prior: bool = True,
        verbose: bool = True,
        algorithm: str = 'pcmci',  # 'pc' or 'pcmci'
        max_lag: int = 5,
        tau_min: int = 1
    ):
        """
        Initialize the causal graph builder.
        
        Args:
            alpha: Significance level for independence tests (default: 0.05)
            use_trace_prior: Whether to use trace data as prior knowledge
            verbose: Whether to print progress information
            algorithm: Algorithm to use - 'pc' (causal-learn) or 'pcmci' (tigramite, recommended for time series)
            max_lag: Maximum time lag for PCMCI+ algorithm (default: 5)
            tau_min: Minimum time lag for PCMCI+ algorithm (default: 1, excludes instantaneous effects)
        """
        self.alpha = alpha
        self.use_trace_prior = use_trace_prior
        self.verbose = verbose
        self.algorithm = algorithm.lower()
        self.max_lag = max_lag
        self.tau_min = tau_min
        
        if self.algorithm not in ['pc', 'pcmci']:
            raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'pc' or 'pcmci'")
        
    def build_causal_graph(
        self,
        data: pd.DataFrame,
        service_topology: Optional[nx.DiGraph] = None,
        background_knowledge: Optional[Any] = None
    ) -> nx.DiGraph:
        """
        Build causal graph from data using selected algorithm (PC or PCMCI+).
        
        Args:
            data: Wide table DataFrame (each row is a time point, each column is a metric)
            service_topology: Optional service topology graph from traces
            background_knowledge: Optional background knowledge (for PC algorithm only)
            
        Returns:
            NetworkX DiGraph representing the causal graph
        """
        if self.algorithm == 'pcmci':
            return self._build_causal_graph_pcmci(data, service_topology)
        else:
            return self._build_causal_graph_pc(data, service_topology, background_knowledge)
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and clean data for causal discovery.
        
        Args:
            data: Raw input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            raise ValueError("Data is empty")
        
        if self.verbose:
            print(f"Building causal graph from {data.shape[0]} time points and {data.shape[1]} features...")
        
        # Prepare data - remove any remaining NaN values
        data_clean = data.dropna(axis=1, how='all')  # Remove columns with all NaN
        data_clean = data_clean.ffill().bfill()
        
        if data_clean.empty:
            raise ValueError("Data is empty after cleaning")
        
        return data_clean
    
    def _build_causal_graph_pc(
        self,
        data: pd.DataFrame,
        service_topology: Optional[nx.DiGraph] = None,
        background_knowledge: Optional[Any] = None
    ) -> nx.DiGraph:
        """
        Build causal graph using PC algorithm from causal-learn.
        
        Args:
            data: Wide table DataFrame (each row is a time point, each column is a metric)
            service_topology: Optional service topology graph from traces
            background_knowledge: Optional background knowledge for PC algorithm
            
        Returns:
            NetworkX DiGraph representing the causal graph
        """
        # Prepare data
        data_clean = self._prepare_data(data)
        
        # Remove constant columns (zero variance) - these cause singular matrix errors
        if self.verbose:
            print("Checking for constant columns...")
        constant_cols = []
        for col in data_clean.columns:
            if data_clean[col].nunique() <= 1 or data_clean[col].std() == 0:
                constant_cols.append(col)
        
        if constant_cols:
            if self.verbose:
                print(f"Removing {len(constant_cols)} constant columns:")
                for col in constant_cols:
                    print(f"  - {col}")
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
        
        # Use iterative method based on correlation matrix condition number
        # This handles multi-collinearity by ensuring the correlation matrix is well-conditioned
        if self.verbose:
            print("Checking for linearly dependent columns using correlation matrix condition number...")
        try:
            temp_data = data_clean.copy()
            max_condition = 1e12
            max_iterations = min(100, data_clean.shape[1] // 2)  # Don't remove too many columns
            removed_cols = []
            
            for iteration in range(max_iterations):
                try:
                    # Compute correlation matrix
                    corr_matrix = temp_data.corr().values
                    cond_num = np.linalg.cond(corr_matrix)
                    
                    if cond_num < max_condition:
                        # Matrix is well-conditioned
                        if iteration > 0 and self.verbose:
                            print(f"  Condition number reduced to {cond_num:.2e} after removing "
                                  f"{len(removed_cols)} columns")
                        break
                    
                    # Find column with highest correlation sum (most correlated with others)
                    abs_corr = np.abs(corr_matrix - np.eye(len(temp_data.columns)))
                    col_sums = np.sum(abs_corr, axis=0)
                    worst_col_idx = np.argmax(col_sums)
                    worst_col = temp_data.columns[worst_col_idx]
                    
                    removed_cols.append(worst_col)
                    if self.verbose:
                        print(f"  Removing {worst_col} (condition number: {cond_num:.2e})")
                    
                    temp_data = temp_data.drop(columns=[worst_col])
                    if temp_data.empty or temp_data.shape[1] < 2:
                        break
                        
                except np.linalg.LinAlgError:
                    # Matrix is singular, need to remove more columns
                    # Try to remove a column based on variance (remove low variance column)
                    try:
                        col_vars = temp_data.var()
                        worst_col = col_vars.idxmin()
                        removed_cols.append(worst_col)
                        if self.verbose:
                            print(f"  Removing {worst_col} (singular matrix, low variance)")
                        temp_data = temp_data.drop(columns=[worst_col])
                        if temp_data.empty or temp_data.shape[1] < 2:
                            break
                    except Exception:
                        break
                except Exception as e:
                    if self.verbose:
                        print(f"  Stopped checking: {e}")
                    break
            
            if removed_cols:
                data_clean = temp_data
                if self.verbose:
                    print(f"Removed {len(removed_cols)} linearly dependent columns")
            else:
                if self.verbose:
                    try:
                        corr_matrix = data_clean.corr().values
                        cond_num = np.linalg.cond(corr_matrix)
                        print(f"  Correlation matrix condition number: {cond_num:.2e} (OK)")
                    except Exception:
                        print(f"  Could not compute condition number")
                    
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not check for linearly dependent columns: {e}")
            # Continue anyway
        
        if data_clean.empty:
            raise ValueError("Data is empty after removing correlated columns")
        
        # Check for remaining issues that could cause singular matrix
        if self.verbose:
            print("Final data check...")
        
        # Convert to numpy array
        data_array = data_clean.values
        
        # Check for NaN or Inf values
        if np.isnan(data_array).any() or np.isinf(data_array).any():
            if self.verbose:
                print("Warning: Data contains NaN or Inf values, replacing...")
            data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Final check: ensure correlation matrix is not singular
        try:
            corr_check = np.corrcoef(data_array.T)
            cond_num = np.linalg.cond(corr_check)
            
            # If condition number is too high, iteratively remove columns
            max_condition = 1e10
            max_iterations = min(50, data_clean.shape[1] // 2)  # Don't remove too many
            
            if cond_num > max_condition:
                if self.verbose:
                    print(f"Warning: Correlation matrix condition number is high ({cond_num:.2e}), "
                          f"iteratively removing problematic columns...")
                
                temp_data = data_clean.copy()
                removed_in_final = []
                
                for iteration in range(max_iterations):
                    try:
                        temp_array = temp_data.values.astype(float)
                        temp_array = np.nan_to_num(temp_array, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Standardize
                        mean = np.mean(temp_array, axis=0)
                        std = np.std(temp_array, axis=0)
                        std[std == 0] = 1.0
                        temp_normalized = (temp_array - mean) / std
                        
                        corr_temp = np.corrcoef(temp_normalized.T)
                        cond_temp = np.linalg.cond(corr_temp)
                        
                        if cond_temp < max_condition:
                            if self.verbose:
                                print(f"  Condition number reduced to {cond_temp:.2e} after removing "
                                      f"{len(removed_in_final)} columns")
                            break
                        
                        # Find column with highest correlation sum (most correlated with others)
                        abs_corr = np.abs(corr_temp - np.eye(len(temp_data.columns)))
                        col_sums = np.sum(abs_corr, axis=0)
                        worst_col_idx = np.argmax(col_sums)
                        worst_col = temp_data.columns[worst_col_idx]
                        
                        removed_in_final.append(worst_col)
                        if self.verbose:
                            print(f"  Removing {worst_col} (high condition number: {cond_temp:.2e})")
                        
                        temp_data = temp_data.drop(columns=[worst_col])
                        if temp_data.empty or temp_data.shape[1] < 2:
                            break
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"  Stopped iterative removal: {e}")
                        break
                
                if removed_in_final:
                    data_clean = temp_data
                    data_array = data_clean.values.astype(float)
                    data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
                    if self.verbose:
                        print(f"Removed {len(removed_in_final)} additional columns in final check")
            else:
                if self.verbose:
                    print(f"Correlation matrix condition number: {cond_num:.2e} (OK)")
                    
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not check correlation matrix condition: {e}")
            # Continue anyway - let PC algorithm handle it
        
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
    
    def _build_causal_graph_pcmci(
        self,
        data: pd.DataFrame,
        service_topology: Optional[nx.DiGraph] = None
    ) -> nx.DiGraph:
        """
        Build causal graph using PCMCI+ algorithm from tigramite.
        PCMCI+ is specifically designed for time series data and handles temporal dependencies better.
        
        Args:
            data: Wide table DataFrame (each row is a time point, each column is a metric)
            service_topology: Optional service topology graph from traces
            
        Returns:
            NetworkX DiGraph representing the causal graph
        """
        # Prepare data
        data_clean = self._prepare_data(data)
        
        # Remove constant columns
        if self.verbose:
            print("Checking for constant columns...")
        constant_cols = []
        for col in data_clean.columns:
            if data_clean[col].nunique() <= 1 or data_clean[col].std() == 0:
                constant_cols.append(col)
        
        if constant_cols:
            if self.verbose:
                print(f"Removing {len(constant_cols)} constant columns:")
                for col in constant_cols:
                    print(f"  - {col}")
            data_clean = data_clean.drop(columns=constant_cols)
        
        if data_clean.empty:
            raise ValueError("Data is empty after removing constant columns")
        
        # Convert to numpy array and handle NaN/Inf
        data_array = data_clean.values.astype(float)
        data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        feature_names = data_clean.columns.tolist()
        num_features = len(feature_names)
        
        if self.verbose:
            print(f"Using {data_array.shape[0]} time points and {num_features} variables")
            print(f"Using PCMCI+ algorithm with max_lag={self.max_lag}, tau_min={self.tau_min}")
        
        # Create tigramite DataFrame
        # tigramite DataFrame expects data in shape (T, N) where T is time, N is number of variables
        dataframe = dp.DataFrame(
            data_array,
            var_names=feature_names
        )
        
        # Build link assumptions from service topology if provided
        link_assumptions = None
        if self.use_trace_prior and service_topology is not None:
            link_assumptions = self._build_link_assumptions(
                feature_names,
                service_topology,
                self.max_lag
            )
            if self.verbose:
                num_forbidden = sum(len(links) for links in link_assumptions.values())
                print(f"Using service topology constraints: {num_forbidden} links forbidden")
        
        # Initialize independence test (Partial Correlation for continuous data)
        cond_ind_test = ParCorr()
        
        # Initialize PCMCI algorithm
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test
        )
        
        # Run PCMCI+ algorithm
        if self.verbose:
            print("Running PCMCI+ algorithm...")
        
        # Prepare run_pcmci arguments
        run_args = {
            'tau_min': self.tau_min,
            'tau_max': self.max_lag,
            'pc_alpha': self.alpha
        }
        
        # Add link_assumptions if provided
        if link_assumptions is not None:
            run_args['link_assumptions'] = link_assumptions
        
        results = pcmci.run_pcmci(**run_args)
        
        # Convert results to NetworkX graph
        causal_graph = self._convert_pcmci_to_networkx(
            results,
            feature_names,
            self.max_lag,
            self.tau_min
        )
        
        if self.verbose:
            print(f"Causal graph: {causal_graph.number_of_nodes()} nodes, "
                  f"{causal_graph.number_of_edges()} edges")
        
        return causal_graph
    
    def _build_link_assumptions(
        self,
        feature_names: List[str],
        service_topology: nx.DiGraph,
        max_lag: int
    ) -> Dict[int, Dict[Tuple[int, int], str]]:
        """
        Build link_assumptions dictionary for PCMCI+ from service topology.
        
        Args:
            feature_names: List of feature names (metric names)
            service_topology: Service topology graph from traces
            max_lag: Maximum time lag
            
        Returns:
            link_assumptions dictionary in format: {j: {(i, -tau): link_type, ...}, ...}
            - link_type: 'o-o' for forbidden, '-->' or '<--' for forced (not used here)
            - j: target variable index
            - i: source variable index  
            - -tau: negative time lag
        """
        num_features = len(feature_names)
        link_assumptions = {}
        
        # Create mapping from feature names to service names
        # Assume feature naming convention: ServiceName_MetricName or ServiceName-MetricName
        feature_to_service = {}
        for feature_name in feature_names:
            # Try to extract service name from feature name
            # Common patterns: "ServiceName_Metric", "ServiceName-Metric", etc.
            if '_' in feature_name:
                parts = feature_name.split('_')
                service_name = parts[0]
            elif '-' in feature_name:
                parts = feature_name.split('-')
                service_name = parts[0]
            else:
                # If no delimiter, use the whole name (may not be accurate)
                service_name = feature_name
            
            # Store mapping (even if service name might not match topology exactly)
            feature_to_service[feature_name] = service_name
        
        # Forbid links between metrics from services that are not connected in topology
        for i, f1 in enumerate(feature_names):
            for j, f2 in enumerate(feature_names):
                if i == j:
                    continue
                
                service1 = feature_to_service.get(f1, None)
                service2 = feature_to_service.get(f2, None)
                
                # If we can't determine service names, skip (let algorithm decide)
                if service1 is None or service2 is None:
                    continue
                
                # If services are the same, allow links (within-service dependencies)
                if service1 == service2:
                    continue
                
                # If services are not connected in topology (in either direction), forbid links
                if not service_topology.has_edge(service1, service2) and \
                   not service_topology.has_edge(service2, service1):
                    # Forbid all lagged links between these metrics
                    # Format: {j: {(i, -tau): 'o-o', ...}, ...}
                    if j not in link_assumptions:
                        link_assumptions[j] = {}
                    
                    # Add forbidden links for all lags from tau_min to max_lag
                    for lag in range(1, max_lag + 1):  # lag 1 to max_lag
                        link_assumptions[j][(i, -lag)] = 'o-o'
        
        return link_assumptions
    
    def _convert_pcmci_to_networkx(
        self,
        results: Dict[str, Any],
        feature_names: List[str],
        max_lag: int,
        tau_min: int
    ) -> nx.DiGraph:
        """
        Convert PCMCI+ results to NetworkX DiGraph.
        
        Args:
            results: Results dictionary from pcmci.run_pcmci()
            feature_names: List of feature names
            max_lag: Maximum time lag used
            tau_min: Minimum time lag used
            
        Returns:
            NetworkX DiGraph representing the causal graph
        """
        causal_graph = nx.DiGraph()
        
        # Add all nodes
        for name in feature_names:
            causal_graph.add_node(name)
        
        # Extract graph and p_matrix from results
        graph = results['graph']  # Shape: (num_features, num_features, max_lag+1)
        p_matrix = results['p_matrix']  # Shape: (num_features, num_features, max_lag+1)
        val_matrix = results.get('val_matrix', None)  # Optional: causal strength
        
        num_features = len(feature_names)
        
        # Process all lagged links (from tau_min to max_lag, excluding instantaneous lag=0)
        for var_idx_from in range(num_features):
            for var_idx_to in range(num_features):
                if var_idx_from == var_idx_to:
                    continue
                
                # Check all lags from tau_min to max_lag
                for lag in range(tau_min, max_lag + 1):
                    # Check if link is significant
                    # 'x' in graph means significant link
                    if graph[var_idx_from, var_idx_to, lag] == 'x':
                        from_node_name = feature_names[var_idx_from]
                        to_node_name = feature_names[var_idx_to]
                        
                        # Get p-value and optional value
                        p_value = p_matrix[var_idx_from, var_idx_to, lag]
                        edge_attrs = {
                            'lag': lag,
                            'p_value': float(p_value) if not np.isnan(p_value) else 1.0
                        }
                        
                        if val_matrix is not None:
                            val = val_matrix[var_idx_from, var_idx_to, lag]
                            if not np.isnan(val):
                                edge_attrs['causal_strength'] = float(val)
                        
                        # Add edge (if not exists) or update if this lag has better p-value
                        if not causal_graph.has_edge(from_node_name, to_node_name):
                            causal_graph.add_edge(from_node_name, to_node_name, **edge_attrs)
                        else:
                            # If edge exists, keep the one with better (lower) p-value
                            current_p = causal_graph[from_node_name][to_node_name].get('p_value', 1.0)
                            if p_value < current_p:
                                causal_graph[from_node_name][to_node_name].update(edge_attrs)
        
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

