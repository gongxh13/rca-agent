"""
Metric Analysis Tool

Provides high-level semantic operations for analyzing time-series metrics data
in the context of root cause analysis. Uses LangChain's @tool decorator for agent integration.

This is the abstract base class. Use LocalMetricAnalysisTool for OpenRCA dataset analysis.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from langchain.tools import tool

from .base import BaseRCATool


class MetricAnalysisTool(BaseRCATool):
    """
    Tool for analyzing time-series metrics with high-level semantic operations.
    
    This tool analyzes both application-level metrics (service performance) and
    infrastructure-level metrics (CPU, memory, disk, network) to support root cause analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metric analysis tool.
        
        Args:
            config: Configuration dictionary that may include:
                   - dataset_path: Path to the dataset
                   - metric_source: Source type (local, remote, etc.)
        """
        super().__init__(config)
        self.metric_source = None
        
    def initialize(self) -> None:
        """Initialize metric data source connections."""
        super().initialize()
        # Subclasses should override to initialize their specific data sources
        
    def get_tools(self) -> List[Any]:
        """Get list of LangChain tools for metric analysis."""
        return [
            self.get_service_performance,
            self.find_slow_services,
            self.find_low_success_rate_services,
            self.detect_metric_anomalies,
            self.get_available_components,
            self.get_available_metrics,
            self.compare_service_metrics,
            self.compare_container_metrics,
            self.get_metric_statistics,
        ]
    
    # Application Metrics Tools (service-level analysis)
    
    @tool
    def compare_service_metrics(
        self,
        start_time: str,
        end_time: str,
        baseline_start: Optional[str] = None,
        baseline_end: Optional[str] = None,
        service_name: Optional[str] = None
    ) -> str:
        """Compare service metrics between a target period and a baseline period.
        
        Analyzes performance metrics (response time, success rate, etc.) for a specific time range
        and compares them against a baseline period to identify deviations and anomalies.
        If baseline period is not provided, it attempts to infer one.
        
        Args:
            start_time: Target period start time (ISO format)
            end_time: Target period end time (ISO format)
            baseline_start: Optional baseline period start time (ISO format)
            baseline_end: Optional baseline period end time (ISO format)
            service_name: Optional specific service to analyze
            
        Returns:
            Detailed comparison report highlighting changes in metrics, including
            percentage changes and robust statistical analysis (outliers removed).
        """
        raise NotImplementedError()

    @tool
    def compare_container_metrics(
        self,
        start_time: str,
        end_time: str,
        component_name: str,
        metric_pattern: Optional[str] = None,
        baseline_start: Optional[str] = None,
        baseline_end: Optional[str] = None
    ) -> str:
        """Compare container/infrastructure metrics between a target period and a baseline period.
        
        Analyzes infrastructure metrics (CPU, Memory, Disk, etc.) for a specific component
        and compares them against a baseline period.
        
        Args:
            start_time: Target period start time (ISO format)
            end_time: Target period end time (ISO format)
            component_name: Name of the component to analyze (e.g., 'Tomcat01')
            metric_pattern: Optional pattern to filter metrics (e.g., 'CPU', 'Memory')
            baseline_start: Optional baseline period start time
            baseline_end: Optional baseline period end time
            
        Returns:
            Comparison report for matching metrics.
        """
        raise NotImplementedError()

    @tool
    def get_metric_statistics(
        self,
        start_time: str,
        end_time: str,
        component_name: str,
        metric_name: str
    ) -> str:
        """Get detailed statistical breakdown for a specific metric (MicroRCA style).
        
        Provides comprehensive statistics including mean, std, min, max,
        percentiles (p25, p50, p75, p95, p99), and non-zero ratios.
        Useful for deep-dive analysis of a specific anomalous metric.
        
        Args:
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            component_name: Name of the component (e.g., 'Tomcat01')
            metric_name: Exact name of the metric (e.g., 'OSLinux-CPU_CPU_CPUCpuUtil')
            
        Returns:
            JSON-formatted string with detailed statistics.
        """
        raise NotImplementedError()

    @tool
    def get_service_performance(
        self,
        start_time: str,
        end_time: str,
        service_name: Optional[str] = None
    ) -> str:
        """Get performance metrics for services in a time range.
        
        Analyzes service-level metrics including response time, success rate,
        and request volume to understand application performance.
        
        Args:
            start_time: Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
            end_time: End time in ISO format
            service_name: Optional specific service to analyze (if None, analyzes all services)
            
        Returns:
            Formatted summary of service performance including average response time,
            success rate, request count, and identification of any concerning patterns
        """
        raise NotImplementedError()
    
    @tool
    def find_slow_services(
        self,
        start_time: str,
        end_time: str,
        threshold_ms: float = 500.0
    ) -> str:
        """Find services with high response times.
        
        Identifies services experiencing slow response times that may indicate
        performance issues or bottlenecks.
        
        Args:
            start_time: Start time in ISO format
            end_time: End time in ISO format
            threshold_ms: Response time threshold in milliseconds (default: 500ms)
            
        Returns:
            List of slow services with their average response times, p95/p99 latencies,
            and comparison to normal baseline
        """
        raise NotImplementedError()
    
    @tool
    def find_low_success_rate_services(
        self,
        start_time: str,
        end_time: str,
        threshold_percent: float = 95.0
    ) -> str:
        """Find services with low success rates.
        
        Identifies services experiencing elevated error rates that may indicate
        failures or degraded functionality.
        
        Args:
            start_time: Start time in ISO format
            end_time: End time in ISO format
            threshold_percent: Success rate threshold percentage (default: 95%)
            
        Returns:
            List of services with low success rates, their actual success percentages,
            and request volumes affected
        """
        raise NotImplementedError()
    
    # Infrastructure Metrics Tools (container/component-level analysis)
    
    @tool
    def detect_metric_anomalies(
        self,
        start_time: str,
        end_time: str,
        method: str = "both",
        component_id: Optional[str] = None,
        sensitivity: float = 3.0,
        top: int = 10,
        ruptures_algorithm: str = "pelt",
        ruptures_model: str = "rbf",
        pen: float = 5.0,
        z_threshold: Optional[float] = None,
        min_data_points_ruptures: int = 10,
        min_data_points_zscore: int = 5,
        min_consecutive: int = 3
    ) -> str:
        """Detect anomalous metric values using ruptures or Z-score methods.
        
        A robust tool for detecting anomalies in core metrics (CPU, memory, disk, network, JVM)
        using either ruptures change point detection or Z-score statistical analysis.
        Automatically handles timezone conversion and focuses on candidate components.
        
        Args:
            start_time: Start time in ISO format (e.g., "2021-03-04T01:00:00" or "2021-03-04T01:00:00+08:00")
            end_time: End time in ISO format (e.g., "2021-03-04T01:30:00" or "2021-03-04T01:30:00+08:00")
            method: Detection method - "ruptures", "zscore", or "both" (default: "both")
            component_id: Optional specific component to analyze (if None, analyzes all candidate components)
            sensitivity: Z-score threshold for zscore method (default: 3.0)
            top: Maximum number of anomalies to return (default: 10, not used if method="both")
            ruptures_algorithm: Algorithm for ruptures - "pelt", "binseg", "dynp", "window" (default: "pelt")
            ruptures_model: Model for ruptures - "rbf", "l1", "l2", "linear", "normal", "ar", "rank" (default: "rbf")
            pen: Penalty parameter for ruptures (default: 5.0)
            z_threshold: Z-score threshold (default: None, uses sensitivity if None)
            min_data_points_ruptures: Minimum data points for ruptures (default: 10)
            min_data_points_zscore: Minimum data points for zscore (default: 5)
            min_consecutive: Minimum consecutive anomaly points for zscore (default: 3)
            
        Returns:
            JSON string containing list of anomalies with:
            - component_name: Component name
            - faulty_kpi: KPI name
            - fault_start_time: ISO format timestamp (fault start point)
            - severity_score: Severity description
            - deviation_pct: Deviation percentage
            - method: Detection method used
            
        Ruptures Algorithms:
            - "pelt": Pruned Exact Linear Time - Fast and accurate, good for most scenarios (default)
            - "binseg": Binary Segmentation - Fast but may not find global optimum
            - "dynp": Dynamic Programming - Global optimum but computationally expensive
            - "window": Window-based - Good for online detection
            
        Ruptures Models:
            - "rbf": Radial Basis Function - Good for non-linear patterns (default)
            - "l1": L1 norm - Robust to outliers
            - "l2": L2 norm - Standard least squares
            - "linear": Linear model - For linear trends
            - "normal": Normal distribution - For Gaussian data
            - "ar": Auto-regressive - For time series with dependencies
            - "rank": Rank-based - Non-parametric, robust
        """
        raise NotImplementedError()
    
    @tool
    def get_available_components(
        self,
        start_time: str,
        end_time: str
    ) -> str:
        """Get list of available component IDs in the specified time range.
        
        Useful for discovering what infrastructure components have metrics data available.
        
        Args:
            start_time: Start time in ISO format
            end_time: End time in ISO format
            
        Returns:
            List of unique component IDs (cmdb_id) that have metric data in the time range
        """
        raise NotImplementedError()
    
    @tool
    def get_available_metrics(
        self,
        start_time: str,
        end_time: str,
        component_id: Optional[str] = None,
        metric_pattern: Optional[str] = None,
        top: int = 10
    ) -> str:
        """Get list of available metric names in the specified time range.
        
        Useful for discovering what metrics are available for analysis.
        
        Args:
            start_time: Start time in ISO format
            end_time: End time in ISO format
            component_id: Optional filter by specific component (if None, returns all metrics)
            metric_pattern: Optional pattern to filter metric names (case-insensitive)
            top: Maximum metrics to show per group (default: 10)
            
        Returns:
            List of unique metric names (kpi_name) available in the time range
        """
        raise NotImplementedError()
    
    def cleanup(self) -> None:
        """Clean up metric data source connections."""
        if self.metric_source:
            self.metric_source = None
        super().cleanup()
