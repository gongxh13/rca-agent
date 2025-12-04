"""
Local Metric Analysis Tool

Concrete implementation of MetricAnalysisTool for the OpenRCA dataset.
Uses local CSV files via OpenRCADataLoader to provide metric analysis.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .metric_tool import MetricAnalysisTool
from .data_loader import OpenRCADataLoader


class LocalMetricAnalysisTool(MetricAnalysisTool):
    """
    Local implementation of MetricAnalysisTool using OpenRCA dataset files.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the local metric tool.
        
        Args:
            config: Configuration dictionary containing:
                   - dataset_path: Path to the OpenRCA dataset root
        """
        super().__init__(config)
        self.data_loader: Optional[OpenRCADataLoader] = None
        
    def initialize(self) -> None:
        """Initialize the data loader."""
        super().initialize()
        dataset_path = self.config.get("dataset_path", "datasets/OpenRCA/Bank")
        default_tz = self.config.get("default_timezone", "Asia/Shanghai")
        self.data_loader = OpenRCADataLoader(dataset_path, default_timezone=default_tz)
        
    def _check_loader(self) -> None:
        """Check if data loader is initialized."""
        if not self.data_loader:
            raise RuntimeError("Tool not initialized. Call initialize() first.")

    # Application Metrics Tools
    
    def get_service_performance(
        self,
        start_time: str,
        end_time: str,
        service_name: Optional[str] = None
    ) -> str:
        self._check_loader()
        df = self.data_loader.load_metrics_for_time_range(start_time, end_time, "app")
        
        if df.empty:
            return f"No application metrics found between {start_time} and {end_time}"
            
        if service_name:
            df = df[df['tc'] == service_name]
            if df.empty:
                return f"No metrics found for service '{service_name}' in the specified time range"
        
        # Calculate statistics
        stats = df.groupby('tc').agg({
            'mrt': ['mean', 'max', 'min'],
            'sr': 'mean',
            'rr': 'mean',
            'cnt': 'sum'
        }).round(2)
        
        # Format output
        output = [f"Service Performance Summary ({start_time} to {end_time}):"]
        
        for service in stats.index:
            s = stats.loc[service]
            output.append(f"\nService: {service}")
            output.append(f"  - Avg Response Time: {s[('mrt', 'mean')]}ms (Range: {s[('mrt', 'min')]}-{s[('mrt', 'max')]}ms)")
            output.append(f"  - Success Rate: {s[('sr', 'mean')]}%")
            output.append(f"  - Request Rate: {s[('rr', 'mean')]}%")
            output.append(f"  - Total Requests: {int(s[('cnt', 'sum')])}")
            
            # Add insights
            if s[('sr', 'mean')] < 99.0:
                output.append(f"  ⚠️ Low success rate detected (<99%)")
            if s[('mrt', 'mean')] > 500:
                output.append(f"  ⚠️ High latency detected (>500ms)")
                
        return "\n".join(output)
    
    def get_available_components(
        self,
        start_time: str,
        end_time: str
    ) -> str:
        self._check_loader()
        df = self.data_loader.load_metrics_for_time_range(start_time, end_time, "container")
        
        if df.empty:
            return "No container metrics found in the specified time range"
        
        components = sorted(df['cmdb_id'].unique())
        
        output = [f"Available Components ({len(components)} total):"]
        for comp in components:
            # Get metric count for this component
            metric_count = df[df['cmdb_id'] == comp]['kpi_name'].nunique()
            output.append(f"  - {comp} ({metric_count} metrics)")
        
        return "\n".join(output)
    
    def get_available_metrics(
        self,
        start_time: str,
        end_time: str,
        component_id: Optional[str] = None,
        metric_pattern: Optional[str] = None,
        top: int = 10
    ) -> str:
        self._check_loader()
        df = self.data_loader.load_metrics_for_time_range(start_time, end_time, "container")
        
        if df.empty:
            return "No container metrics found in the specified time range"
        
        if component_id:
            df = df[df['cmdb_id'] == component_id]
            if df.empty:
                return f"No metrics found for component '{component_id}'"
        
        # Apply metric pattern filter if provided
        if metric_pattern:
            df = df[df['kpi_name'].str.contains(metric_pattern, case=False, na=False)]
            if df.empty:
                return f"No metrics matching pattern '{metric_pattern}' found"
        
        metrics = sorted(df['kpi_name'].unique())
        
        # Build header
        if component_id and metric_pattern:
            output = [f"Available Metrics for {component_id} matching '{metric_pattern}' ({len(metrics)} total):"]
        elif component_id:
            output = [f"Available Metrics for {component_id} ({len(metrics)} total):"]
        elif metric_pattern:
            output = [f"Available Metrics matching '{metric_pattern}' ({len(metrics)} total):"]
        else:
            output = [f"Available Metrics (All Components, {len(metrics)} total):"]
        
        # Group metrics by prefix for better readability
        metric_groups = {}
        for metric in metrics:
            # Extract prefix (e.g., "OSLinux-CPU" from "OSLinux-CPU_CPU_CPUUtil")
            parts = metric.split('_')
            if len(parts) > 1:
                prefix = parts[0]
            else:
                prefix = "Other"
            
            if prefix not in metric_groups:
                metric_groups[prefix] = []
            metric_groups[prefix].append(metric)
        
        # Display grouped metrics
        for prefix in sorted(metric_groups.keys()):
            output.append(f"\n{prefix} ({len(metric_groups[prefix])} metrics):")
            for metric in sorted(metric_groups[prefix])[:top]:  # Use top parameter
                output.append(f"  - {metric}")
            if len(metric_groups[prefix]) > top:
                output.append(f"  ... and {len(metric_groups[prefix]) - top} more")
        
        return "\n".join(output)

    def find_slow_services(
        self,
        start_time: str,
        end_time: str,
        threshold_ms: float = 500.0
    ) -> str:
        self._check_loader()
        df = self.data_loader.load_metrics_for_time_range(start_time, end_time, "app")
        
        if df.empty:
            return "No data found for analysis"
            
        # Group by service and calculate mean response time
        avg_rt = df.groupby('tc')['mrt'].mean()
        slow_services = avg_rt[avg_rt > threshold_ms].sort_values(ascending=False)
        
        if slow_services.empty:
            return f"No services exceeded the latency threshold of {threshold_ms}ms"
            
        output = [f"Slow Services Report (Threshold: {threshold_ms}ms):"]
        for service, rt in slow_services.items():
            # Get peak latency for this service
            peak = df[df['tc'] == service]['mrt'].max()
            output.append(f"\n🔴 {service}")
            output.append(f"  - Average Latency: {rt:.2f}ms")
            output.append(f"  - Peak Latency: {peak:.2f}ms")
            
        return "\n".join(output)

    def find_low_success_rate_services(
        self,
        start_time: str,
        end_time: str,
        threshold_percent: float = 95.0
    ) -> str:
        self._check_loader()
        df = self.data_loader.load_metrics_for_time_range(start_time, end_time, "app")
        
        if df.empty:
            return "No data found for analysis"
            
        avg_sr = df.groupby('tc')['sr'].mean()
        problem_services = avg_sr[avg_sr < threshold_percent].sort_values()
        
        if problem_services.empty:
            return f"All services operating above {threshold_percent}% success rate"
            
        output = [f"Low Success Rate Report (Threshold: {threshold_percent}%):"]
        for service, sr in problem_services.items():
            min_sr = df[df['tc'] == service]['sr'].min()
            output.append(f"\n🔴 {service}")
            output.append(f"  - Average Success Rate: {sr:.2f}%")
            output.append(f"  - Minimum Success Rate: {min_sr:.2f}%")
            
        return "\n".join(output)

    def compare_service_performance(
        self,
        service_name: str,
        current_start: str,
        current_end: str,
        baseline_start: str,
        baseline_end: str
    ) -> str:
        self._check_loader()
        
        # Load data for both periods
        curr_df = self.data_loader.load_metrics_for_time_range(current_start, current_end, "app")
        base_df = self.data_loader.load_metrics_for_time_range(baseline_start, baseline_end, "app")
        
        if curr_df.empty or base_df.empty:
            return "Insufficient data for comparison"
            
        # Filter for service
        curr_svc = curr_df[curr_df['tc'] == service_name]
        base_svc = base_df[base_df['tc'] == service_name]
        
        if curr_svc.empty or base_svc.empty:
            return f"Service '{service_name}' not found in one or both time ranges"
            
        # Calculate metrics
        metrics = {
            'Response Time': ('mrt', 'mean'),
            'Success Rate': ('sr', 'mean'),
            'Request Count': ('cnt', 'sum')
        }
        
        output = [f"Performance Comparison for {service_name}:"]
        output.append(f"Baseline: {baseline_start} to {baseline_end}")
        output.append(f"Current:  {current_start} to {current_end}\n")
        
        for label, (col, agg) in metrics.items():
            base_val = base_svc[col].agg(agg)
            curr_val = curr_svc[col].agg(agg)
            
            if base_val == 0:
                pct_change = 0 if curr_val == 0 else 100
            else:
                pct_change = ((curr_val - base_val) / base_val) * 100
            
            change_symbol = "↑" if pct_change > 0 else "↓"
            if abs(pct_change) < 1: change_symbol = "="
            
            output.append(f"{label}:")
            output.append(f"  Baseline: {base_val:.2f}")
            output.append(f"  Current:  {curr_val:.2f}")
            output.append(f"  Change:   {change_symbol} {abs(pct_change):.1f}%")
            
        return "\n".join(output)

    # Infrastructure Metrics Tools
    
    def get_resource_metrics(
        self,
        component_id: str,
        metric_pattern: str,
        start_time: str,
        end_time: str
    ) -> str:
        self._check_loader()
        df = self.data_loader.load_metrics_for_time_range(start_time, end_time, "container")
        
        if df.empty:
            return "No infrastructure metrics found"
            
        # Filter by component and metric pattern
        df = df[df['cmdb_id'] == component_id]
        if df.empty:
            return f"No metrics found for component '{component_id}'"
            
        # Case-insensitive match for metric pattern
        df = df[df['kpi_name'].str.contains(metric_pattern, case=False, na=False)]
        if df.empty:
            return f"No metrics matching '{metric_pattern}' found for {component_id}"
            
        # Group by KPI name
        stats = df.groupby('kpi_name')['value'].agg(['min', 'max', 'mean', 'std']).round(2)
        
        output = [f"Resource Metrics for {component_id} matching '{metric_pattern}':"]
        for kpi in stats.index:
            s = stats.loc[kpi]
            output.append(f"\n{kpi}:")
            output.append(f"  - Avg: {s['mean']}")
            output.append(f"  - Max: {s['max']}")
            output.append(f"  - Min: {s['min']}")
            output.append(f"  - Std Dev: {s['std']}")
            
        return "\n".join(output)

    def find_high_resource_usage(
        self,
        metric_pattern: str,
        start_time: str,
        end_time: str,
        threshold: float = 80.0,
        top: int = 10
    ) -> str:
        self._check_loader()
        df = self.data_loader.load_metrics_for_time_range(start_time, end_time, "container")
        
        if df.empty:
            return "No data found"
            
        # Filter metrics by pattern (case-insensitive)
        df = df[df['kpi_name'].str.contains(metric_pattern, case=False, na=False)]
        
        if df.empty:
            return f"No metrics found matching pattern '{metric_pattern}'"
        
        # Find components exceeding threshold
        high_usage = []
        for (comp, kpi), group in df.groupby(['cmdb_id', 'kpi_name']):
            max_val = group['value'].max()
            if max_val > threshold:
                avg_val = group['value'].mean()
                high_usage.append({
                    'component': comp,
                    'kpi': kpi,
                    'max': max_val,
                    'avg': avg_val
                })
        
        if not high_usage:
            return f"No components found with metrics matching '{metric_pattern}' exceeding {threshold}"
            
        # Sort by max usage
        high_usage.sort(key=lambda x: x['max'], reverse=True)
        
        output = [f"High Resource Usage Report (Pattern: '{metric_pattern}', Threshold: > {threshold}):"]
        for item in high_usage[:top]:  # Limit to top N
            output.append(f"\n🔴 {item['component']}")
            output.append(f"  Metric: {item['kpi']}")
            output.append(f"  Peak Value: {item['max']:.2f}")
            output.append(f"  Avg Value:  {item['avg']:.2f}")
            
        return "\n".join(output)

    def detect_metric_anomalies(
        self,
        start_time: str,
        end_time: str,
        component_id: Optional[str] = None,
        sensitivity: float = 3.0,
        top: int = 10
    ) -> str:
        self._check_loader()
        df = self.data_loader.load_metrics_for_time_range(start_time, end_time, "container")
        
        if df.empty:
            return "No data found"
            
        if component_id:
            df = df[df['cmdb_id'] == component_id]
            
        anomalies = []
        
        # Analyze each KPI for each component
        # Optimization: Filter out constant values first
        for (comp, kpi), group in df.groupby(['cmdb_id', 'kpi_name']):
            values = group['value']
            if values.nunique() <= 1:
                continue
                
            mean = values.mean()
            std = values.std()
            
            if std == 0:
                continue
                
            # Z-score detection
            z_scores = np.abs((values - mean) / std)
            anomaly_mask = z_scores > sensitivity
            
            if anomaly_mask.any():
                num_anomalies = anomaly_mask.sum()
                max_z = z_scores.max()
                anomalies.append({
                    'component': comp,
                    'kpi': kpi,
                    'count': num_anomalies,
                    'max_z': max_z,
                    'mean': mean,
                    'std': std
                })
        
        if not anomalies:
            return "No anomalies detected"
            
        # Sort by severity (max z-score)
        anomalies.sort(key=lambda x: x['max_z'], reverse=True)
        
        output = [f"Anomaly Detection Report (Sensitivity: {sensitivity} std devs):"]
        for item in anomalies[:top]:  # Use top parameter
            output.append(f"\n⚠️ {item['component']} - {item['kpi']}")
            output.append(f"  Anomalies Found: {item['count']} data points")
            output.append(f"  Max Deviation: {item['max_z']:.1f}x standard deviation")
            output.append(f"  Baseline: {item['mean']:.2f} ± {item['std']:.2f}")
            
        return "\n".join(output)

    def get_component_health_summary(
        self,
        start_time: str,
        end_time: str,
        component_id: Optional[str] = None,
        metric_pattern: Optional[str] = None,
        warning_threshold: float = 80.0,
        critical_threshold: float = 90.0
    ) -> str:
        self._check_loader()
        df = self.data_loader.load_metrics_for_time_range(start_time, end_time, "container")
        
        if df.empty:
            return "No data found"
            
        if component_id:
            df = df[df['cmdb_id'] == component_id]
            if df.empty:
                return f"No data found for component '{component_id}'"
        
        # Apply metric pattern filter if provided
        if metric_pattern:
            df = df[df['kpi_name'].str.contains(metric_pattern, case=False, na=False)]
            if df.empty:
                pattern_msg = f" matching pattern '{metric_pattern}'" if metric_pattern else ""
                return f"No metrics found{pattern_msg}"
        
        summary = {}
        
        for comp, comp_df in df.groupby('cmdb_id'):
            status = "Healthy"
            issues = []
            
            # Check each metric against thresholds
            for kpi, kpi_df in comp_df.groupby('kpi_name'):
                max_val = kpi_df['value'].max()
                
                if max_val > critical_threshold:
                    status = "Critical"
                    issues.append(f"{kpi}: {max_val:.1f} (critical)")
                elif max_val > warning_threshold:
                    if status != "Critical":
                        status = "Warning"
                    issues.append(f"{kpi}: {max_val:.1f} (warning)")
            
            summary[comp] = {'status': status, 'issues': issues}
        
        # Build output
        pattern_info = f" (Pattern: '{metric_pattern}')" if metric_pattern else ""
        output = [f"Component Health Summary{pattern_info}:"]
        output.append(f"Thresholds: Warning > {warning_threshold}, Critical > {critical_threshold}")
        
        # Group by status
        for status_type in ["Critical", "Warning", "Healthy"]:
            comps = {k: v for k, v in summary.items() if v['status'] == status_type}
            if comps:
                icon = "🔴" if status_type == "Critical" else "⚠️" if status_type == "Warning" else "✅"
                output.append(f"\n{icon} {status_type} ({len(comps)} components):")
                for comp, details in comps.items():
                    if details['issues']:
                        # Show top 3 issues per component to avoid too much output
                        top_issues = details['issues'][:3]
                        issue_str = "\n    " + "\n    ".join(top_issues)
                        if len(details['issues']) > 3:
                            issue_str += f"\n    ... and {len(details['issues']) - 3} more issues"
                        output.append(f"  - {comp}:{issue_str}")
                    else:
                        output.append(f"  - {comp}")
                    
        return "\n".join(output)
