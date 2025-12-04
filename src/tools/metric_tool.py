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
            self.compare_service_performance,
            self.get_resource_metrics,
            self.find_high_resource_usage,
            self.detect_metric_anomalies,
            self.get_component_health_summary,
            self.get_available_components,
            self.get_available_metrics,
        ]
    
    # Application Metrics Tools (service-level analysis)
    
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
    
    @tool
    def compare_service_performance(
        self,
        service_name: str,
        current_start: str,
        current_end: str,
        baseline_start: str,
        baseline_end: str
    ) -> str:
        """Compare service performance between current and baseline periods.
        
        Helps identify if current service behavior deviates from normal patterns
        by comparing against a baseline period.
        
        Args:
            service_name: Name of the service to analyze
            current_start: Start of current period in ISO format
            current_end: End of current period in ISO format
            baseline_start: Start of baseline period in ISO format
            baseline_end: End of baseline period in ISO format
            
        Returns:
            Comparison showing percentage changes in response time, success rate,
            and request volume between the two periods
        """
        raise NotImplementedError()
    
    # Infrastructure Metrics Tools (container/component-level analysis)
    
    @tool
    def get_resource_metrics(
        self,
        component_id: str,
        metric_pattern: str,
        start_time: str,
        end_time: str
    ) -> str:
        """Get specific resource metrics for a component.
        
        Retrieves and summarizes infrastructure metrics like CPU, memory, disk I/O,
        or network for a specific component.
        
        Args:
            component_id: Component identifier (e.g., "Tomcat01", "Mysql02")
            metric_pattern: Pattern to match metric names (e.g., "CPU", "Memory", "Disk")
            start_time: Start time in ISO format
            end_time: End time in ISO format
            
        Returns:
            Statistical summary of matching metrics including min, max, average,
            and p95 values with trend analysis
        """
        raise NotImplementedError()
    
    @tool
    def find_high_resource_usage(
        self,
        metric_pattern: str,
        start_time: str,
        end_time: str,
        threshold: float = 80.0,
        top: int = 10
    ) -> str:
        """Find components with high resource usage based on metric pattern.
        
        Identifies infrastructure components with metrics exceeding a threshold.
        Useful for finding resource bottlenecks and stressed components.
        
        Args:
            metric_pattern: Pattern to match metric names (e.g., "CPU", "Memory", "Disk", "Network")
            start_time: Start time in ISO format
            end_time: End time in ISO format
            threshold: Threshold value for filtering (default: 80.0). Unit depends on the metric.
            top: Maximum number of results to return (default: 10)
            
        Returns:
            List of top components exceeding the threshold with their peak and average
            values, sorted by peak value
        """
        raise NotImplementedError()
    
    @tool
    def detect_metric_anomalies(
        self,
        start_time: str,
        end_time: str,
        component_id: Optional[str] = None,
        sensitivity: float = 3.0,
        top: int = 10
    ) -> str:
        """Detect anomalous metric values using statistical analysis.
        
        Uses z-score based anomaly detection to find unusual metric values
        that deviate significantly from normal patterns.
        
        Args:
            start_time: Start time in ISO format
            end_time: End time in ISO format
            component_id: Optional specific component to analyze (if None, analyzes all)
            sensitivity: Number of standard deviations for anomaly threshold (default: 3.0)
            top: Maximum number of anomalies to return (default: 10)
            
        Returns:
            List of detected anomalies with metric names, component IDs, timestamps,
            values, and severity scores
        """
        raise NotImplementedError()
    
    @tool
    def get_component_health_summary(
        self,
        start_time: str,
        end_time: str,
        component_id: Optional[str] = None,
        metric_pattern: Optional[str] = None,
        warning_threshold: float = 80.0,
        critical_threshold: float = 90.0
    ) -> str:
        """Get health summary for infrastructure components based on metric thresholds.
        
        Provides an overall health assessment of components by checking if metrics
        exceed warning or critical thresholds.
        
        Args:
            start_time: Start time in ISO format
            end_time: End time in ISO format
            component_id: Optional specific component (if None, summarizes all components)
            metric_pattern: Optional pattern to filter which metrics to check (e.g., "CPU", "Memory")
                          If None, checks all metrics
            warning_threshold: Threshold for warning status (default: 80.0)
            critical_threshold: Threshold for critical status (default: 90.0)
            
        Returns:
            Health summary with status (healthy/warning/critical) for each component,
            key metrics, and any concerning patterns detected
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
