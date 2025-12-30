"""
Root Cause Inference Module

This module provides functionality for identifying root causes of anomalies
using DoWhy-GCM based on trained Structural Causal Models (SCM).
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional, Dict, List, Any, Tuple, Union
import json

try:
    import dowhy.gcm as gcm
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False


class RootCauseAnalyzer:
    """
    Analyzer for root cause inference using DoWhy-GCM.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the root cause analyzer.
        
        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        
        if not DOWHY_AVAILABLE:
            raise ImportError(
                "DoWhy library is not available. "
                "Please install it with: pip install dowhy"
            )
            
    def analyze(
        self,
        model: Any,
        target_node: str,
        anomaly_data: pd.DataFrame,
        n_top: int = 5
    ) -> Dict[str, Any]:
        """
        Perform root cause analysis for a specific target node and anomaly data.
        
        Args:
            model: Trained GCM model
            target_node: The node where the anomaly was observed (e.g., service response time)
            anomaly_data: DataFrame containing the anomalous data (single row or multiple rows)
            n_top: Number of top root causes to return
            
        Returns:
            Dictionary containing root cause analysis results
        """
        if self.verbose:
            print(f"Analyzing root causes for target '{target_node}' with {len(anomaly_data)} samples...")
            
        # Validate target node exists in model
        if target_node not in model.graph.nodes:
            raise ValueError(f"Target node '{target_node}' not found in the causal model")
            
        # Perform anomaly attribution
        # attribute_anomalies returns a dictionary: {node: attribution_score}
        # The score represents the contribution of each node to the anomaly in the target node
        try:
            attribution_scores = gcm.attribute_anomalies(
                model,
                target_node,
                anomaly_data
            )
        except Exception as e:
            raise RuntimeError(f"Error during anomaly attribution: {e}")
            
        # Process and sort results
        # Convert to standard dictionary and sort by absolute score (magnitude of contribution)
        # Note: DoWhy-GCM returns shapley values, which can be negative. 
        # Large absolute values indicate strong influence.
        scores = {}
        for node, score in attribution_scores.items():
            try:
                arr = np.array(score, dtype=float).reshape(-1)
                if arr.size == 0:
                    val = 0.0
                else:
                    val = float(np.nanmean(arr))
            except Exception:
                try:
                    val = float(score)
                except Exception:
                    val = 0.0
            scores[node] = val
        
        # Sort by absolute value in descending order
        sorted_causes = sorted(
            scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Format the results
        top_causes = []
        for node, score in sorted_causes[:n_top]:
            # Try to parse component and metric from node name
            component = "Unknown"
            metric = node
            if '_' in node:
                parts = node.split('_', 1)
                component = parts[0]
                metric = parts[1]
                
            top_causes.append({
                'node': node,
                'component': component,
                'metric': metric,
                'score': score,
                'contribution': abs(score) # Use absolute value for ranking
            })
            
        result = {
            'target_node': target_node,
            'root_causes': top_causes,
            'all_scores': scores
        }
        
        if self.verbose:
            print(f"Top root cause: {top_causes[0]['node']} (score: {top_causes[0]['score']:.4f})")
            
        return result

    def batch_analyze(
        self,
        model: Any,
        target_nodes: List[str],
        anomaly_data: pd.DataFrame,
        n_top: int = 5
    ) -> Dict[str, Any]:
        """
        Perform root cause analysis for multiple target nodes.
        
        Args:
            model: Trained GCM model
            target_nodes: List of target nodes
            anomaly_data: DataFrame containing the anomalous data
            n_top: Number of top root causes to return per target
            
        Returns:
            Dictionary containing aggregated results
        """
        results = {}
        
        for target in target_nodes:
            try:
                results[target] = self.analyze(model, target, anomaly_data, n_top)
            except Exception as e:
                print(f"Error analyzing target {target}: {e}")
                results[target] = {'error': str(e)}
                
        return results
