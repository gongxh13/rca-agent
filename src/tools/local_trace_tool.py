"""
Local Trace Analysis Tool

Implementation of TraceAnalysisTool for OpenRCA dataset.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx

from .trace_tool import TraceAnalysisTool
from .data_loader import OpenRCADataLoader
from ..utils.time_utils import to_iso_shanghai

class LocalTraceAnalysisTool(TraceAnalysisTool):
    """
    Local implementation of TraceAnalysisTool using OpenRCA dataset files.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.loader = None
        
    def initialize(self) -> None:
        """Initialize the data loader."""
        dataset_path = self.config.get("dataset_path", "datasets/OpenRCA/Bank")
        default_tz = self.config.get("default_timezone", "Asia/Shanghai")
        self.loader = OpenRCADataLoader(dataset_path, default_timezone=default_tz)
        
    def find_slow_spans(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        service_name: Optional[str] = None,
        min_duration_ms: int = 1000,
        limit: int = 10
    ) -> str:
        """Find the slowest spans in the specified time range."""
        if not start_time or not end_time:
            return "Error: start_time and end_time are required for local analysis"
            
        df = self.loader.load_traces_for_time_range(start_time, end_time)
        if df.empty:
            return "No trace data found for the specified time range."
            
        # Filter by duration
        slow_spans = df[df['duration'] >= min_duration_ms].copy()
        
        # Filter by service name if provided
        if service_name:
            slow_spans = slow_spans[slow_spans['cmdb_id'] == service_name]
            
        if slow_spans.empty:
            return f"No spans found with duration >= {min_duration_ms}ms" + (f" for service {service_name}" if service_name else "")
            
        # Sort by duration descending
        slow_spans = slow_spans.sort_values('duration', ascending=False).head(limit)
        
        result = [f"Top {len(slow_spans)} slow spans (>= {min_duration_ms}ms):"]
        for _, row in slow_spans.iterrows():
            result.append(
                f"- Service: {row['cmdb_id']}, Duration: {row['duration']}ms, "
                f"TraceID: {row['trace_id']}, SpanID: {row['span_id']}, "
                f"Time: {to_iso_shanghai(row['datetime'])}"
            )
            
        return "\n".join(result)

    def analyze_call_chain(
        self,
        trace_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> str:
        """Analyze the call chain for specific traces."""
        if not trace_id:
            return "Error: trace_id is required for local analysis"
            
        # If start/end time not provided, we might need to search a wider range or require them
        # For now, let's assume the user provides them or we default to a reasonable window if we could,
        # but since we don't know when the trace happened without querying, we really need time range.
        # However, to be helpful, if start_time is missing, we might fail.
        if not start_time or not end_time:
             return "Error: start_time and end_time are required to locate the trace efficiently"

        df = self.loader.load_traces_for_time_range(start_time, end_time)
        if df.empty:
            return "No trace data found for the specified time range."
            
        trace_spans = df[df['trace_id'] == trace_id].copy()
        
        if trace_spans.empty:
            return f"Trace ID {trace_id} not found in the specified time range."
            
        # Build tree
        # We can use networkx to build the graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        # Map span_id to row data for easy access
        span_map = {}
        for _, row in trace_spans.iterrows():
            span_id = row['span_id']
            span_map[span_id] = row
            G.add_node(span_id, service=row['cmdb_id'], duration=row['duration'])
            
            parent_id = row['parent_id']
            # Check if parent exists in this trace (it might be a root span or parent might be missing/external)
            # In this dataset, root spans might have a specific parent_id format or just not exist in the map
            if parent_id in span_map: # This assumes we process parent first or we add edge later. 
                # Better to add edges after processing all nodes or just add edge if parent is known
                pass

        # Second pass for edges
        for _, row in trace_spans.iterrows():
            span_id = row['span_id']
            parent_id = row['parent_id']
            if parent_id in span_map and parent_id != span_id:
                G.add_edge(parent_id, span_id)
                
        # Find root(s)
        roots = [n for n, d in G.in_degree() if d == 0]
        
        if not roots:
            # If still no roots, we might have a real cycle A->B->A
            # We can try to break cycles or just pick the node with the earliest timestamp
            # For now, let's try to find cycles and break them or just pick the earliest node
            try:
                cycles = list(nx.simple_cycles(G))
                if cycles:
                    # Pick the node in the cycle with the earliest timestamp as the root
                    # This is a heuristic
                    all_nodes = set()
                    for cycle in cycles:
                        all_nodes.update(cycle)
                    
                    # Find node with min timestamp in these cycles
                    root = min(all_nodes, key=lambda n: span_map[n]['timestamp'])
                    roots = [root]
            except:
                pass
                
        if not roots:
             # Fallback: pick the node with the earliest timestamp
             root = min(span_map.keys(), key=lambda n: span_map[n]['timestamp'])
             roots = [root]
             result = [f"Warning: Cycle detected or root ambiguous. Using earliest span as root."]
        else:
             result = [f"Call Chain for Trace {trace_id}:"]
        
        for root in roots:
            tree_edges = nx.bfs_edges(G, root)
            # Add root first
            root_data = span_map[root]
            result.append(f"[{root_data['cmdb_id']}] {root_data['duration']}ms (Root)")
            
            # Traverse
            # To print nicely with indentation, we might want a recursive function instead of simple edge list
            self._print_tree(G, root, span_map, result, level=1)
            
        return "\n".join(result)
        
    def _print_tree(self, G, node, span_map, result, level):
        children = sorted(G.successors(node), key=lambda x: span_map[x]['timestamp'])
        for child in children:
            data = span_map[child]
            indent = "  " * level
            result.append(f"{indent}└─ [{data['cmdb_id']}] {data['duration']}ms")
            self._print_tree(G, child, span_map, result, level + 1)

    def get_service_dependencies(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        service_name: Optional[str] = None
    ) -> str:
        """Get service dependency graph from trace data."""
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
            
        df = self.loader.load_traces_for_time_range(start_time, end_time)
        if df.empty:
            return "No trace data found."
            
        # Build dependency graph: Parent Service -> Child Service
        # We need to join the dataframe with itself on parent_id = span_id to get parent service
        
        # Create a mapping of span_id -> service
        # Note: This might be memory intensive for large datasets. 
        # Optimization: Select only needed columns
        spans = df[['span_id', 'cmdb_id', 'parent_id']].copy()
        
        # Self join to match parent_id with span_id
        # left: child (current row), right: parent
        merged = pd.merge(
            spans, 
            spans, 
            left_on='parent_id', 
            right_on='span_id', 
            suffixes=('_child', '_parent'),
            how='inner'
        )
        
        # Group by parent_service, child_service
        deps = merged.groupby(['cmdb_id_parent', 'cmdb_id_child']).size().reset_index(name='count')
        
        if service_name:
            # Filter where service is either parent or child
            deps = deps[
                (deps['cmdb_id_parent'] == service_name) | 
                (deps['cmdb_id_child'] == service_name)
            ]
            
        if deps.empty:
            return "No dependencies found."
            
        result = ["Service Dependencies:"]
        for _, row in deps.iterrows():
            result.append(f"{row['cmdb_id_parent']} -> {row['cmdb_id_child']} (Calls: {row['count']})")
            
        return "\n".join(result)

    def detect_latency_anomalies(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        service_name: Optional[str] = None,
        sensitivity: float = 0.8
    ) -> str:
        """Detect anomalous latency patterns."""
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
            
        df = self.loader.load_traces_for_time_range(start_time, end_time)
        if df.empty:
            return "No trace data found."
            
        if service_name:
            df = df[df['cmdb_id'] == service_name]
            
        if df.empty:
            return "No data for specified service."
            
        # Calculate stats per service
        stats = df.groupby('cmdb_id')['duration'].agg(['mean', 'std', 'count']).reset_index()
        
        # Filter out services with too few samples
        stats = stats[stats['count'] > 10]
        
        anomalies = []
        
        # For each span, check if it's anomalous (z-score)
        # We can merge stats back to df
        df_merged = pd.merge(df, stats, on='cmdb_id')
        
        # Z-score = (x - mean) / std
        # Sensitivity 0.8 -> roughly 2-3 std devs? 
        # Let's map sensitivity 0.0-1.0 to Z-score threshold 5.0-2.0
        # 1.0 -> 2.0 (very sensitive), 0.0 -> 5.0 (least sensitive)
        z_threshold = 5.0 - (sensitivity * 3.0)
        
        df_merged['z_score'] = (df_merged['duration'] - df_merged['mean']) / df_merged['std']
        
        anomalous_spans = df_merged[df_merged['z_score'] > z_threshold].sort_values('z_score', ascending=False)
        
        if anomalous_spans.empty:
            return "No latency anomalies detected."
            
        result = [f"Detected {len(anomalous_spans)} latency anomalies (Threshold Z-Score > {z_threshold:.1f}):"]
        
        # Show top 10
        top_anomalies = anomalous_spans.head(10)
        for _, row in top_anomalies.iterrows():
            result.append(
                f"- Service: {row['cmdb_id']}, Duration: {row['duration']}ms "
                f"(Mean: {row['mean']:.1f}, Z: {row['z_score']:.1f}), "
                f"TraceID: {row['trace_id']}"
            )
            
        return "\n".join(result)

    def find_failed_traces(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        service_name: Optional[str] = None,
        limit: int = 10
    ) -> str:
        """Find traces that contain errors."""
        # Note: The current dataset schema doesn't seem to have an error code or status column.
        # We will return a message stating this limitation, but provide a hook for future extension.
        return (
            "Analysis of failed traces is currently limited as the dataset does not explicitly "
            "indicate error status (e.g., HTTP status codes) in the trace spans. "
            "Future versions may infer failures from missing spans or specific tag patterns."
        )

    def identify_bottlenecks(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        min_impact_percentage: float = 10.0
    ) -> str:
        """Identify performance bottlenecks."""
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
            
        df = self.loader.load_traces_for_time_range(start_time, end_time)
        if df.empty:
            return "No trace data found."
            
        # Calculate total duration per service
        total_duration = df['duration'].sum()
        service_duration = df.groupby('cmdb_id')['duration'].sum().reset_index()
        service_duration['impact'] = (service_duration['duration'] / total_duration) * 100
        
        bottlenecks = service_duration[service_duration['impact'] >= min_impact_percentage].sort_values('impact', ascending=False)
        
        if bottlenecks.empty:
            return f"No single service contributes more than {min_impact_percentage}% to total system latency."
            
        result = ["Identified Bottlenecks (Services consuming significant time):"]
        for _, row in bottlenecks.iterrows():
            result.append(
                f"- {row['cmdb_id']}: {row['impact']:.1f}% of total time "
                f"({row['duration']}ms total)"
            )
            
        return "\n".join(result)
