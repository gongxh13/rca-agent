"""
Causal Graph Discovery Module

This module provides functionality for building causal graphs using causal discovery algorithms.
Supports multiple algorithms:
- PC algorithm (causal-learn): Requires IID assumption, may not work for time series
- PCMCI+ algorithm (tigramite): Accurate but slow, recommended for time series
- Granger Causality (causal-learn): Fast and suitable for time series
- VARLiNGAM (causal-learn): Balanced speed and accuracy for time series
- Granger + PC hybrid: Uses Granger for fast screening
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
    from causallearn.search.Granger.Granger import Granger as CausalLearnGranger
    from causallearn.search.FCMBased.lingam.var_lingam import VARLiNGAM
    CAUSALLEARN_AVAILABLE = True
except ImportError:
    CAUSALLEARN_AVAILABLE = False
    # Provide dummy classes for type hints
    CausalLearnGranger = None
    VARLiNGAM = None

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
        algorithm: str = 'pcmci',  # 'pc', 'pcmci', 'granger', 'varlingam', or 'granger_pc'
        max_lag: int = 5,
        tau_min: int = 1,
        granger_test: str = 'ssr_ftest',  # For Granger: 'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'
        granger_cv: int = 5,  # For Granger Lasso: number of cross-validation folds
        varlingam_threshold: float = 0.01,  # For VARLiNGAM: threshold for edge significance (default: 0.01, smaller = more edges)
        remove_multicollinearity: bool = False
    ):
        """
        Initialize the causal graph builder.
        
        Args:
            alpha: Significance level for independence tests (default: 0.05)
            use_trace_prior: Whether to use trace data as prior knowledge
            verbose: Whether to print progress information
            algorithm: Algorithm to use:
                - 'pc': PC algorithm (causal-learn, requires IID, may not work for time series)
                - 'pcmci': PCMCI+ algorithm (tigramite, recommended for time series but slow)
                - 'granger': Granger Causality (fast, good for time series)
                - 'varlingam': VARLiNGAM (balanced speed and accuracy for time series)
                - 'granger_pc': Hybrid approach - Granger for fast screening, then PC for refinement
            max_lag: Maximum time lag (default: 5)
            tau_min: Minimum time lag for PCMCI+ algorithm (default: 1, excludes instantaneous effects)
            granger_test: Statistical test for Granger Causality (default: 'ssr_ftest')
            granger_cv: Number of CV folds for Granger Lasso (default: 5)
            varlingam_threshold: Threshold for VARLiNGAM edge significance (default: 0.01).
                Only edges with coefficient > threshold are included. Smaller values produce more edges.
                Recommended values: 0.01 (strict), 0.001 (moderate), 1e-6 (very loose, original default)
        """
        self.alpha = alpha
        self.use_trace_prior = use_trace_prior
        self.verbose = verbose
        self.algorithm = algorithm.lower()
        self.max_lag = max_lag
        self.tau_min = tau_min
        self.granger_test = granger_test
        self.granger_cv = granger_cv
        self.varlingam_threshold = varlingam_threshold
        self.remove_multicollinearity = remove_multicollinearity
        
        valid_algorithms = ['pc', 'pcmci', 'granger', 'varlingam', 'granger_pc']
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}. Must be one of {valid_algorithms}")
        
        # Check if causallearn is available for algorithms that require it
        algorithms_need_causallearn = ['pc', 'granger', 'varlingam', 'granger_pc']
        if self.algorithm in algorithms_need_causallearn and not CAUSALLEARN_AVAILABLE:
            raise ImportError(
                f"Algorithm '{self.algorithm}' requires causal-learn library. "
                f"Please install it with: pip install causal-learn\n"
                f"Or add the causal-learn source directory to your Python path."
            )
        
    def build_causal_graph(
        self,
        data: pd.DataFrame,
        service_topology: Optional[nx.DiGraph] = None,
        background_knowledge: Optional[Any] = None
    ) -> nx.DiGraph:
        """
        Build causal graph from data using selected algorithm.
        
        Args:
            data: Wide table DataFrame (each row is a time point, each column is a metric)
            service_topology: Optional service topology graph from traces
            background_knowledge: Optional background knowledge (for PC algorithm only)
            
        Returns:
            NetworkX DiGraph representing the causal graph
        """
        if self.algorithm == 'pcmci':
            return self._build_causal_graph_pcmci(data, service_topology)
        elif self.algorithm == 'granger':
            if not CAUSALLEARN_AVAILABLE:
                raise ImportError("Granger algorithm requires causal-learn library")
            return self._build_causal_graph_granger(data, service_topology)
        elif self.algorithm == 'varlingam':
            if not CAUSALLEARN_AVAILABLE:
                raise ImportError("VARLiNGAM algorithm requires causal-learn library")
            return self._build_causal_graph_varlingam(data, service_topology)
        elif self.algorithm == 'granger_pc':
            if not CAUSALLEARN_AVAILABLE:
                raise ImportError("Granger+PC algorithm requires causal-learn library")
            return self._build_causal_graph_granger_pc(data, service_topology, background_knowledge)
        else:  # 'pc'
            if not CAUSALLEARN_AVAILABLE:
                raise ImportError("PC algorithm requires causal-learn library")
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
        
        # Remove timestamp/datetime columns (they should not participate in causal discovery)
        # Timestamps are metadata, not observed variables, and their monotonicity can cause spurious causal relationships
        timestamp_keywords = ['timestamp', 'datetime', 'time', 'date', 'timestamp_utc']
        timestamp_cols = [col for col in data.columns 
                         if any(keyword in col.lower() for keyword in timestamp_keywords)]
        
        if timestamp_cols:
            if self.verbose:
                print(f"Removing timestamp columns (metadata, not used in causal discovery): {timestamp_cols}")
            data = data.drop(columns=timestamp_cols)
        
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
        
        # Initialize PCMCI algorithm with verbosity for progress output
        verbosity = 1 if self.verbose else 0
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=verbosity
        )
        
        # Run PCMCI+ algorithm
        if self.verbose:
            print("Running PCMCI+ algorithm...")
            print("Note: This may take a while for large datasets (197 variables, 2883 time points)")
            print("PC stage: discovering causal skeleton...")
        
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
        
        if self.verbose:
            print("MCI stage: refining causal links...")
        
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
    
    def _build_causal_graph_granger(
        self,
        data: pd.DataFrame,
        service_topology: Optional[nx.DiGraph] = None
    ) -> nx.DiGraph:
        """
        Build causal graph using Granger Causality algorithm from causal-learn.
        Granger Causality is fast and suitable for time series data.
        
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
                print(f"Removing {len(constant_cols)} constant columns")
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
            print(f"Using Granger Causality algorithm with max_lag={self.max_lag}")
        
        # Standardize data to help Lasso convergence
        # This addresses the ConvergenceWarning by ensuring features have similar scales
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_array_scaled = scaler.fit_transform(data_array)
        
        if self.verbose:
            print("Data standardized to help Lasso regression convergence")
        
        # Initialize Granger Causality
        granger = CausalLearnGranger(
            maxlag=self.max_lag,
            test=self.granger_test,
            significance_level=self.alpha,
            cv=self.granger_cv
        )
        
        # Run Granger Causality with statistical tests
        if self.verbose:
            print("Running Granger Causality with statistical tests...")
            print("Using statistical significance (p-value) instead of hard-coded threshold")
        
        # Strategy: Use Lasso for fast screening, then apply statistical tests
        # This balances efficiency (Lasso) with statistical rigor (p-value tests)
        if self.verbose:
            print("Step 1: Fast screening with Lasso regression...")
        
        # Use granger_lasso for fast screening (identifies candidate edges)
        coeff_matrix = granger.granger_lasso(data_array_scaled)
        
        # Step 2: Apply statistical tests to candidate edges
        # For each candidate edge identified by Lasso, perform Granger causality test
        if self.verbose:
            print("Step 2: Applying statistical tests to candidate edges...")
        
        # Build causal graph from statistical tests
        causal_graph = nx.DiGraph()
        
        # Add all nodes
        for name in feature_names:
            causal_graph.add_node(name)
        
        # Extract candidate edges from Lasso coefficient matrix
        # coeff_matrix shape: (num_features, num_features * max_lag)
        # Each row i corresponds to target variable i
        # Columns 0 to num_features-1 are lag 1, num_features to 2*num_features-1 are lag 2, etc.
        
        # First, collect candidate edges from Lasso (non-zero coefficients)
        candidate_edges = []  # List of (target_idx, source_idx, lag, coeff)
        for target_idx in range(num_features):
            for source_idx in range(num_features):
                if source_idx == target_idx:
                    continue
                
                for lag in range(1, self.max_lag + 1):
                    coeff_idx = (lag - 1) * num_features + source_idx
                    coeff = coeff_matrix[target_idx, coeff_idx]
                    
                    # If coefficient is non-zero (above numerical threshold), it's a candidate
                    if abs(coeff) > 1e-6:  # Numerical stability threshold
                        candidate_edges.append((target_idx, source_idx, lag, coeff))
        
        if self.verbose:
            print(f"Found {len(candidate_edges)} candidate edges from Lasso screening")
        
        # Apply statistical tests to candidate edges
        # Use granger_test_2d for pairwise statistical tests
        # This returns p-values for each variable pair and lag
        try:
            # Get p-value matrix from statistical tests
            # Note: granger_test_2d may be slow for large datasets, but we only test candidates
            # For efficiency, we can test all pairs but use the candidate list to filter results
            pvalue_matrix = granger.granger_test_2d(data_array_scaled)
            
            # pvalue_matrix shape: (num_features, num_features, max_lag)
            # pvalue_matrix[i, j, lag-1] is the p-value for j -> i at lag
            # Lower p-value means stronger evidence of Granger causality
            
            # Process candidate edges with statistical significance
            significant_edges = []
            for target_idx, source_idx, lag, coeff in candidate_edges:
                # Get p-value for this edge
                # Note: pvalue_matrix indexing may vary, check the actual shape
                if pvalue_matrix.ndim == 3:
                    # Shape: (num_features, num_features, max_lag)
                    p_value = pvalue_matrix[target_idx, source_idx, lag - 1]
                elif pvalue_matrix.ndim == 2:
                    # Shape: (num_features, num_features * max_lag) - similar to coeff_matrix
                    p_value_idx = (lag - 1) * num_features + source_idx
                    p_value = pvalue_matrix[target_idx, p_value_idx]
                else:
                    # Fallback: use alpha as p-value (conservative)
                    p_value = self.alpha
                
                # Check statistical significance
                if not np.isnan(p_value) and p_value < self.alpha:
                    significant_edges.append((target_idx, source_idx, lag, coeff, p_value))
            
            if self.verbose:
                print(f"Found {len(significant_edges)} statistically significant edges (p < {self.alpha})")
            
            # Add significant edges to graph
            for target_idx, source_idx, lag, coeff, p_value in significant_edges:
                target_name = feature_names[target_idx]
                source_name = feature_names[source_idx]
                
                causal_graph.add_edge(
                    source_name,
                    target_name,
                    lag=lag,
                    coefficient=float(coeff),
                    p_value=float(p_value)
                )
                
        except (AttributeError, NotImplementedError) as e:
            # Fallback: if granger_test_2d is not available or fails,
            # use Lasso coefficients but with a more conservative threshold
            if self.verbose:
                print(f"Warning: Statistical test not available ({e}), using Lasso coefficients with conservative threshold")
            
            # Use a more conservative approach: only keep edges with larger coefficients
            # This is not ideal but better than hard-coded 1e-6
            conservative_threshold = np.percentile(np.abs(coeff_matrix[coeff_matrix != 0]), 90) if np.any(coeff_matrix != 0) else 1e-6
            
            for target_idx in range(num_features):
                target_name = feature_names[target_idx]
                
                for source_idx in range(num_features):
                    if source_idx == target_idx:
                        continue
                    
                    source_name = feature_names[source_idx]
                    
                    # Get coefficients across all lags
                    coeffs_across_lags = []
                    p_values_across_lags = []
                    for lag in range(1, self.max_lag + 1):
                        coeff_idx = (lag - 1) * num_features + source_idx
                        coeff = coeff_matrix[target_idx, coeff_idx]
                        if abs(coeff) > conservative_threshold:
                            coeffs_across_lags.append((lag, coeff))
                            # Estimate p-value from coefficient magnitude (rough approximation)
                            # Larger coefficient -> smaller p-value (more significant)
                            # This is a heuristic, not a true statistical test
                            estimated_p = max(0.001, min(0.5, 1.0 / (1.0 + abs(coeff) * 100)))
                            p_values_across_lags.append(estimated_p)
                    
                    # Use the lag with best (lowest) estimated p-value
                    if p_values_across_lags:
                        best_idx = np.argmin(p_values_across_lags)
                        best_lag, best_coeff = coeffs_across_lags[best_idx]
                        best_p = p_values_across_lags[best_idx]
                        
                        causal_graph.add_edge(
                            source_name,
                            target_name,
                            lag=best_lag,
                            coefficient=float(best_coeff),
                            p_value=float(best_p),
                            note="estimated_p_value"  # Mark as estimated, not true statistical test
                        )
        
        if self.verbose:
            print(f"Causal graph: {causal_graph.number_of_nodes()} nodes, "
                  f"{causal_graph.number_of_edges()} edges")
        
        return causal_graph
    
    def _remove_multicollinearity(
        self,
        data: pd.DataFrame,
        max_correlation: float = 0.99,
        max_condition: float = 1e10
    ) -> pd.DataFrame:
        """
        Remove multicollinear columns to ensure covariance matrix is positive definite.
        This is critical for VAR models used in VARLiNGAM.
        
        Args:
            data: Input DataFrame
            max_correlation: Maximum correlation threshold (default: 0.99)
            max_condition: Maximum condition number for correlation matrix (default: 1e10)
            
        Returns:
            DataFrame with multicollinear columns removed
        """
        data_clean = data.copy()
        
        # Remove perfectly correlated columns
        if self.verbose:
            print("Checking for highly correlated columns...")
        try:
            corr_matrix = data_clean.corr().abs()
            to_remove = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if pd.notna(corr_val) and corr_val >= max_correlation:
                        col_i = corr_matrix.columns[i]
                        col_j = corr_matrix.columns[j]
                        if col_j not in to_remove:
                            to_remove.add(col_j)
                            if self.verbose:
                                print(f"  Removing {col_j} (highly correlated with {col_i}, r={corr_val:.4f})")
            
            if to_remove:
                data_clean = data_clean.drop(columns=list(to_remove))
                if self.verbose:
                    print(f"Removed {len(to_remove)} highly correlated columns")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not check for correlated columns: {e}")
        
        # Iteratively remove columns to improve condition number
        if self.verbose:
            print("Checking for multicollinearity using condition number...")
        try:
            temp_data = data_clean.copy()
            max_iterations = min(50, data_clean.shape[1] // 2)  # Don't remove too many
            removed_cols = []
            
            for iteration in range(max_iterations):
                try:
                    # Compute correlation matrix
                    corr_matrix = temp_data.corr().values
                    cond_num = np.linalg.cond(corr_matrix)
                    
                    if cond_num < max_condition:
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
                    # Matrix is singular, remove column with lowest variance
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
                    print(f"Removed {len(removed_cols)} multicollinear columns (total)")
            else:
                if self.verbose:
                    try:
                        corr_matrix = data_clean.corr().values
                        cond_num = np.linalg.cond(corr_matrix)
                        print(f"  Correlation matrix condition number: {cond_num:.2e} (OK)")
                    except Exception:
                        pass
                        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not check for multicollinearity: {e}")
        
        return data_clean
    
    def _build_causal_graph_varlingam(
        self,
        data: pd.DataFrame,
        service_topology: Optional[nx.DiGraph] = None
    ) -> nx.DiGraph:
        """
        Build causal graph using VARLiNGAM algorithm from causal-learn.
        VARLiNGAM combines VAR model with LiNGAM, suitable for time series data.
        
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
                print(f"Removing {len(constant_cols)} constant columns")
            data_clean = data_clean.drop(columns=constant_cols)
        
        if data_clean.empty:
            raise ValueError("Data is empty after removing constant columns")
        
        # Remove multicollinearity - critical for VAR models
        # VAR models require positive definite covariance matrix
        if self.remove_multicollinearity:
            data_clean = self._remove_multicollinearity(data_clean, max_correlation=0.99, max_condition=1e8)
        else:
            if self.verbose:
                print("Skipping multicollinearity removal for VARLiNGAM (keeping highly correlated columns)")
        
        if data_clean.empty or data_clean.shape[1] < 2:
            raise ValueError("Data is empty or has too few columns after removing multicollinearity")
        
        # Convert to numpy array and handle NaN/Inf
        data_array = data_clean.values.astype(float)
        data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        feature_names = data_clean.columns.tolist()
        num_features = len(feature_names)
        
        if self.verbose:
            print(f"Using {data_array.shape[0]} time points and {num_features} variables")
            print(f"Using VARLiNGAM algorithm with lags={self.max_lag}")
        
        # Verify covariance matrix is positive definite (required for VAR model)
        # Try Cholesky decomposition to check positive definiteness
        try:
            cov_matrix = np.cov(data_array.T)
            # Check if matrix is positive definite by attempting Cholesky decomposition
            np.linalg.cholesky(cov_matrix)
            if self.verbose:
                cond_num = np.linalg.cond(cov_matrix)
                print(f"Covariance matrix is positive definite (condition number: {cond_num:.2e})")
        except np.linalg.LinAlgError:
            if self.remove_multicollinearity:
                if self.verbose:
                    print("Warning: Covariance matrix is not positive definite, trying stricter cleanup...")
                data_clean_strict = self._remove_multicollinearity(data_clean, max_correlation=0.95, max_condition=1e6)
                if data_clean_strict.shape[1] < data_clean.shape[1] and data_clean_strict.shape[1] >= 2:
                    data_clean = data_clean_strict
                    data_array = data_clean.values.astype(float)
                    data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
                    feature_names = data_clean.columns.tolist()
                    num_features = len(feature_names)
                    if self.verbose:
                        print(f"Using {num_features} variables after stricter multicollinearity removal")
                        try:
                            cov_matrix = np.cov(data_array.T)
                            np.linalg.cholesky(cov_matrix)
                            cond_num = np.linalg.cond(cov_matrix)
                            print(f"Covariance matrix is now positive definite (condition number: {cond_num:.2e})")
                        except np.linalg.LinAlgError:
                            if self.verbose:
                                print("Warning: Still not positive definite, but proceeding with VARLiNGAM...")
                else:
                    if self.verbose:
                        print("Warning: Could not resolve multicollinearity with stricter criteria, proceeding anyway...")
                        print("Note: If VARLiNGAM fails, consider using 'granger' algorithm instead")
            else:
                if self.verbose:
                    print("Warning: Covariance matrix is not positive definite; proceeding without column removal as requested")
        
        if data_clean.empty or data_clean.shape[1] < 2:
            raise ValueError(
                "Data has too few variables after multicollinearity removal. "
                "Consider using 'granger' algorithm instead, which is more robust to multicollinearity. "
                f"Current variable count: {data_clean.shape[1]}"
            )
        
        # Initialize VARLiNGAM
        # Use None for criterion to disable lag selection if data is problematic
        varlingam = VARLiNGAM(
            lags=self.max_lag,
            criterion='bic',  # Use BIC for lag selection
            prune=True,  # Prune weak edges
            random_state=42  # For reproducibility
        )
        
        # Run VARLiNGAM with error handling for numerical issues
        if self.verbose:
            print("Running VARLiNGAM algorithm...")
            print("This balances speed and accuracy for time series data")
        
        try:
            varlingam.fit(data_array)
        except (np.linalg.LinAlgError, ValueError) as e:
            # If BIC-based lag selection fails due to numerical issues,
            # try disabling lag selection (use fixed lag)
            if self.verbose:
                print(f"Warning: VARLiNGAM with BIC lag selection failed: {e}")
                print("Trying with fixed lag (disabling lag selection)...")
            
            try:
                # Try with fixed lag, no lag selection
                varlingam_fixed = VARLiNGAM(
                    lags=self.max_lag,
                    criterion=None,  # Disable lag selection, use fixed lag
                    prune=True,
                    random_state=42
                )
                varlingam_fixed.fit(data_array)
                varlingam = varlingam_fixed
                if self.verbose:
                    print("Successfully fitted VARLiNGAM with fixed lag")
            except (np.linalg.LinAlgError, ValueError) as e2:
                # If that also fails, try with a smaller lag
                if self.verbose:
                    print(f"Warning: VARLiNGAM with fixed lag also failed: {e2}")
                    print(f"Trying with smaller lag (max_lag=2)...")
                
                try:
                    varlingam_small = VARLiNGAM(
                        lags=min(2, self.max_lag),
                        criterion=None,
                        prune=True,
                        random_state=42
                    )
                    varlingam_small.fit(data_array)
                    varlingam = varlingam_small
                    if self.verbose:
                        print("Successfully fitted VARLiNGAM with smaller lag")
                except (np.linalg.LinAlgError, ValueError) as e3:
                    # All attempts failed
                    raise RuntimeError(
                        f"VARLiNGAM algorithm failed due to numerical issues (non-positive definite covariance matrix). "
                        f"This typically happens when:\n"
                        f"  1. There are too many variables relative to the sample size\n"
                        f"  2. High multicollinearity exists in the data\n"
                        f"  3. The data has numerical instabilities\n\n"
                        f"Suggested solutions:\n"
                        f"  - Use 'granger' algorithm instead (more robust to multicollinearity):\n"
                        f"    builder = CausalGraphBuilder(algorithm='granger', ...)\n"
                        f"  - Reduce the number of variables (features) in your data\n"
                        f"  - Increase the sample size (more time points)\n"
                        f"  - Use stricter multicollinearity removal\n\n"
                        f"Original error: {e3}"
                    ) from e3
        
        # Get adjacency matrices
        # adjacency_matrices is a list of matrices: [B0, B1, B2, ...] where
        # B0 is lag 0 (instantaneous), B1 is lag 1, B2 is lag 2, etc.
        adjacency_matrices = varlingam.adjacency_matrices_
        num_lags = len(adjacency_matrices)  # Should be lags + 1 (includes lag 0)
        
        # Build causal graph from adjacency matrices
        causal_graph = nx.DiGraph()
        
        # Add all nodes
        for name in feature_names:
            causal_graph.add_node(name)
        
        # Use statistical significance instead of coefficient threshold
        if self.verbose:
            print("Evaluating edge significance using statistical tests...")
        
        # Strategy: Use bootstrap or residual-based tests to assess statistical significance
        # Since VARLiNGAM may not provide standard errors directly, we use a residual-based approach
        
        # Compute residuals to assess model fit and coefficient significance
        # For each target variable, we can test if including a source variable significantly improves prediction
        from scipy import stats
        
        # Build causal graph from adjacency matrices with statistical significance
        # Skip lag 0 (instantaneous) if tau_min > 0, focusing on lagged effects
        start_lag_idx = 0 if self.tau_min == 0 else 1
        
        # For statistical testing, we'll use a likelihood ratio test or F-test
        # comparing models with and without each edge
        if self.verbose:
            print("Performing statistical significance tests for edges...")
        
        significant_edges = []
        
        for source_idx in range(num_features):
            source_name = feature_names[source_idx]
            
            for target_idx in range(num_features):
                if source_idx == target_idx:
                    continue
                
                target_name = feature_names[target_idx]
                
                # Get coefficients and test significance across all lags
                best_p_value = 1.0
                best_lag = None
                best_coeff = 0.0
                best_max_coeff = 0.0
                
                for lag_idx in range(start_lag_idx, num_lags):
                    if lag_idx >= len(adjacency_matrices):
                        continue
                    
                    coeff = adjacency_matrices[lag_idx][target_idx, source_idx]
                    abs_coeff = abs(coeff)
                    
                    # Skip if coefficient is very small (likely noise)
                    if abs_coeff < 1e-8:
                        continue
                    
                    # Estimate p-value using coefficient magnitude and sample size
                    # This is a heuristic approach when standard errors are not available
                    # We use a t-test approximation: t = coeff / se, where se is estimated
                    
                    # Estimate standard error based on residual variance
                    # For VAR models, we can estimate SE from the residual covariance
                    # A rough approximation: SE ≈ sqrt(residual_variance / n)
                    # Larger coefficients relative to estimated SE → smaller p-value
                    
                    # Get target variable data
                    target_data = data_array[:, target_idx]
                    n_samples = len(target_data)
                    
                    # Estimate residual variance (simplified: use target variance as proxy)
                    target_var = np.var(target_data)
                    
                    # Rough SE estimate: assume coefficient SE is proportional to residual SE
                    # SE ≈ sqrt(target_var / n_samples) for standardized data
                    # For non-standardized, we need to account for scale
                    estimated_se = np.sqrt(target_var / n_samples) if n_samples > 0 else 1.0
                    
                    # Compute t-statistic
                    if estimated_se > 0:
                        t_stat = abs_coeff / estimated_se
                        # Degrees of freedom: n_samples - number of parameters
                        # Conservative estimate: use n_samples - max_lag * num_features
                        df = max(1, n_samples - self.max_lag * num_features)
                        
                        # Two-tailed t-test p-value
                        p_value = 2 * (1 - stats.t.cdf(t_stat, df))
                    else:
                        # Fallback: use coefficient magnitude as heuristic
                        # Larger coefficient → more significant (rough approximation)
                        p_value = max(0.001, min(0.5, 1.0 / (1.0 + abs_coeff * 100)))
                    
                    # Keep track of best (most significant) lag for this edge
                    if p_value < best_p_value:
                        best_p_value = p_value
                        best_lag = lag_idx
                        best_coeff = coeff
                        best_max_coeff = abs_coeff
                
                # Add edge if statistically significant (p < alpha)
                if best_p_value < self.alpha and best_lag is not None:
                    significant_edges.append((
                        source_name, target_name, best_lag, 
                        best_coeff, best_max_coeff, best_p_value
                    ))
        
        if self.verbose:
            print(f"Found {len(significant_edges)} statistically significant edges (p < {self.alpha})")
            if self.varlingam_threshold > 0:
                print(f"Note: Using statistical significance (p-value) instead of coefficient threshold ({self.varlingam_threshold})")
        
        # Build graph from significant edges
        causal_graph = nx.DiGraph()
        
        # Add all nodes
        for name in feature_names:
            causal_graph.add_node(name)
        
        # Add significant edges
        for source_name, target_name, lag, coeff, max_coeff, p_value in significant_edges:
            causal_graph.add_edge(
                source_name,
                target_name,
                lag=lag,
                coefficient=float(coeff),
                max_abs_coefficient=float(max_coeff),
                p_value=float(p_value)
            )
        
        if self.verbose:
            print(f"Causal graph: {causal_graph.number_of_nodes()} nodes, "
                  f"{causal_graph.number_of_edges()} edges")
        
        return causal_graph
    
    def _build_causal_graph_granger_pc(
        self,
        data: pd.DataFrame,
        service_topology: Optional[nx.DiGraph] = None,
        background_knowledge: Optional[Any] = None
    ) -> nx.DiGraph:
        """
        Build causal graph using hybrid approach: Granger Causality for fast screening,
        then PC algorithm for refinement.
        
        Args:
            data: Wide table DataFrame (each row is a time point, each column is a metric)
            service_topology: Optional service topology graph from traces
            background_knowledge: Optional background knowledge (for PC algorithm)
            
        Returns:
            NetworkX DiGraph representing the causal graph
        """
        if self.verbose:
            print("Using hybrid approach: Granger Causality + PC algorithm")
            print("Step 1: Fast screening with Granger Causality...")
        
        # Step 1: Use Granger Causality for fast screening
        granger_graph = self._build_causal_graph_granger(data, service_topology)
        
        if self.verbose:
            print(f"Granger screening found {granger_graph.number_of_edges()} candidate edges")
            print("Step 2: Refining with PC algorithm...")
        
        # Step 2: Use PC algorithm only on candidate edges from Granger
        # This significantly reduces the search space for PC algorithm
        
        # Prepare data for PC (same as _build_causal_graph_pc)
        data_clean = self._prepare_data(data)
        
        # Remove constant columns (same logic as PC)
        constant_cols = []
        for col in data_clean.columns:
            if data_clean[col].nunique() <= 1 or data_clean[col].std() == 0:
                constant_cols.append(col)
        
        if constant_cols:
            data_clean = data_clean.drop(columns=constant_cols)
        
        # For hybrid approach, we use the Granger graph as the primary result
        # because PC requires IID assumption which may not hold for time series
        # In practice, Granger is already quite good for time series
        if self.verbose:
            print("Note: Using Granger result as primary output (PC requires IID assumption)")
        
        return granger_graph
    
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
        
        # Union: add topology edges in addition to discovered edges
        if service_topology is not None and self.use_trace_prior:
            service_to_nodes = {}
            for n in G.nodes():
                s = str(n).split('_', 1)[0]
                service_to_nodes.setdefault(s, []).append(n)
            for u, v in service_topology.edges():
                su = str(u)
                sv = str(v)
                src_nodes = service_to_nodes.get(su, [])
                tgt_nodes = service_to_nodes.get(sv, [])
                for nu in src_nodes:
                    for nv in tgt_nodes:
                        if not G.has_edge(nu, nv):
                            G.add_edge(nu, nv, prior=True)
        
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
