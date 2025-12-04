"""
Local Log Analysis Tool

Implementation of LogAnalysisTool for OpenRCA dataset.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import re
from collections import Counter
from datetime import datetime

from .log_tool import LogAnalysisTool
from .data_loader import OpenRCADataLoader

class LocalLogAnalysisTool(LogAnalysisTool):
    """
    Local implementation of LogAnalysisTool using OpenRCA dataset files.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.loader = None
        
    def initialize(self) -> None:
        """Initialize the data loader."""
        dataset_path = self.config.get("dataset_path", "datasets/OpenRCA/Bank")
        default_tz = self.config.get("default_timezone", "Asia/Shanghai")
        self.loader = OpenRCADataLoader(dataset_path, default_timezone=default_tz)
        
    def find_error_patterns(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        service_name: Optional[str] = None,
        min_occurrences: int = 3
    ) -> str:
        """Find recurring error patterns in logs."""
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
            
        df = self.loader.load_logs_for_time_range(start_time, end_time)
        if df.empty:
            return "No log data found."
            
        if service_name:
            df = df[df['cmdb_id'] == service_name]
            
        if df.empty:
            return f"No logs found for service {service_name}."
            
        # Simple pattern matching: look for "error", "exception", "fail" case-insensitive
        # In a real system, we might use clustering or more advanced template extraction
        error_logs = df[df['value'].str.contains('error|exception|fail', case=False, na=False)].copy()
        
        if error_logs.empty:
            return "No obvious error patterns found (checked for 'error', 'exception', 'fail')."
            
        # Mask timestamps and numbers to find patterns
        # 1. Mask ISO timestamps
        error_logs['pattern'] = error_logs['value'].str.replace(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{4}', '<TIMESTAMP>', regex=True)
        # 2. Mask other common timestamp formats (e.g. 3748839.786)
        error_logs['pattern'] = error_logs['pattern'].str.replace(r'\d+\.\d+:', '<TIME>:', regex=True)
        # 3. Mask numbers (e.g. sizes, durations)
        error_logs['pattern'] = error_logs['pattern'].str.replace(r'\b\d+\b', '<NUM>', regex=True)
        # 4. Mask hex IDs (simple heuristic)
        error_logs['pattern'] = error_logs['pattern'].str.replace(r'\b[0-9a-fA-F]{8,}\b', '<ID>', regex=True)
        
        patterns = error_logs['pattern'].value_counts()
        
        # Filter by min_occurrences
        patterns = patterns[patterns >= min_occurrences]
        
        if patterns.empty:
            return f"No error patterns found with at least {min_occurrences} occurrences."
            
        result = [f"Found {len(patterns)} recurring error patterns:"]
        for msg, count in patterns.head(10).items():
            result.append(f"- Count: {count}, Pattern: {msg[:200]}...") # Truncate long messages
            
        return "\n".join(result)

    def get_log_summary(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        service_name: Optional[str] = None
    ) -> str:
        """Get a high-level summary of log activity."""
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
            
        df = self.loader.load_logs_for_time_range(start_time, end_time)
        if df.empty:
            return "No log data found."
            
        if service_name:
            df = df[df['cmdb_id'] == service_name]
            
        total_logs = len(df)
        services = df['cmdb_id'].nunique()
        
        # Count errors
        error_count = df['value'].str.contains('error|exception|fail', case=False, na=False).sum()
        warning_count = df['value'].str.contains('warn', case=False, na=False).sum()
        
        top_services = df['cmdb_id'].value_counts().head(5)
        
        result = [
            "Log Summary:",
            f"- Total Entries: {total_logs}",
            f"- Unique Services: {services}",
            f"- Error Count: {error_count} ({(error_count/total_logs)*100:.1f}%)",
            f"- Warning Count: {warning_count}",
            "\nTop Active Services:"
        ]
        
        for svc, count in top_services.items():
            result.append(f"- {svc}: {count} entries")
            
        return "\n".join(result)

    def detect_anomalies(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        sensitivity: float = 0.8
    ) -> str:
        """Detect anomalous log patterns or volumes."""
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
            
        df = self.loader.load_logs_for_time_range(start_time, end_time)
        if df.empty:
            return "No log data found."
            
        # Group by service and time bucket (e.g., 1 minute)
        # Resample requires datetime index
        df_resampled = df.set_index('datetime').groupby('cmdb_id').resample('1T').size().reset_index(name='count')
        
        anomalies = []
        
        # For each service, check for volume spikes
        for service in df_resampled['cmdb_id'].unique():
            svc_data = df_resampled[df_resampled['cmdb_id'] == service]
            if len(svc_data) < 5:
                continue
                
            mean = svc_data['count'].mean()
            std = svc_data['count'].std()
            
            if std == 0:
                continue
                
            # Z-score threshold based on sensitivity
            z_threshold = 5.0 - (sensitivity * 3.0)
            
            svc_data = svc_data.copy()
            svc_data['z_score'] = (svc_data['count'] - mean) / std
            
            spikes = svc_data[svc_data['z_score'] > z_threshold]
            
            for _, row in spikes.iterrows():
                anomalies.append({
                    'service': service,
                    'time': row['datetime'],
                    'count': row['count'],
                    'z_score': row['z_score'],
                    'mean': mean
                })
                
        if not anomalies:
            return "No log volume anomalies detected."
            
        # Sort by z-score
        anomalies.sort(key=lambda x: x['z_score'], reverse=True)
        
        result = [f"Detected {len(anomalies)} log volume anomalies:"]
        for a in anomalies[:10]:
            result.append(
                f"- {a['service']} at {a['time']}: {a['count']} logs/min "
                f"(Mean: {a['mean']:.1f}, Z-Score: {a['z_score']:.1f})"
            )
            
        return "\n".join(result)

    def analyze_error_frequency(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        group_by: str = "service"
    ) -> str:
        """Analyze error frequency grouped by specified dimension."""
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
            
        df = self.loader.load_logs_for_time_range(start_time, end_time)
        if df.empty:
            return "No log data found."
            
        # Filter for errors
        error_df = df[df['value'].str.contains('error|exception|fail', case=False, na=False)]
        
        if error_df.empty:
            return "No errors found in the specified time range."
            
        if group_by == "service":
            counts = error_df['cmdb_id'].value_counts()
            title = "Error Frequency by Service:"
        elif group_by == "host":
             # Assuming cmdb_id maps to host in this dataset
            counts = error_df['cmdb_id'].value_counts()
            title = "Error Frequency by Host:"
        else:
            return f"Unsupported grouping dimension: {group_by}"
            
        result = [title]
        for item, count in counts.head(10).items():
            result.append(f"- {item}: {count} errors")
            
        return "\n".join(result)

    def find_correlated_events(
        self,
        reference_event: str,
        time_window_seconds: int = 300,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> str:
        """Find events that are temporally correlated with a reference event."""
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
            
        df = self.loader.load_logs_for_time_range(start_time, end_time)
        if df.empty:
            return "No log data found."
            
        # Find reference events
        ref_events = df[df['value'].str.contains(reference_event, case=False, na=False)]
        
        if ref_events.empty:
            return f"Reference event pattern '{reference_event}' not found."
            
        correlated_counts = Counter()
        
        # For each reference event, look at window before and after
        for _, ref_row in ref_events.iterrows():
            ref_time = ref_row['datetime']
            window_start = ref_time - pd.Timedelta(seconds=time_window_seconds)
            window_end = ref_time + pd.Timedelta(seconds=time_window_seconds)
            
            # Find events in window
            window_events = df[
                (df['datetime'] >= window_start) & 
                (df['datetime'] <= window_end) &
                (df.index != ref_row.name) # Exclude self
            ]
            
            # Count occurrences of other log messages (simplified by taking first 50 chars)
            # In reality, we'd want better template matching
            for _, row in window_events.iterrows():
                # Skip if it's the same as reference event
                if reference_event.lower() in row['value'].lower():
                    continue
                    
                # Use a simplified key: service + first 30 chars of message
                key = f"[{row['cmdb_id']}] {row['value'][:50]}..."
                correlated_counts[key] += 1
                
        if not correlated_counts:
            return "No correlated events found."
            
        # Top correlated events
        result = [f"Events correlated with '{reference_event}' (Window: +/- {time_window_seconds}s):"]
        for event, count in correlated_counts.most_common(10):
            result.append(f"- Count: {count}, Event: {event}")
            
        return "\n".join(result)

    def query_logs(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        service_name: Optional[str] = None,
        pattern: Optional[str] = None,
        limit: int = 20
    ) -> str:
        """Query and view raw log entries.
        
        This tool allows the model to view actual log content for detailed analysis.
        Useful for investigating specific time periods or examining logs matching a pattern.
        
        Args:
            start_time: Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
            end_time: End time in ISO format
            service_name: Optional filter by service name (cmdb_id)
            pattern: Optional regex pattern to match in log content (case-insensitive)
            limit: Maximum number of log entries to return (default: 20)
            
        Returns:
            Formatted string containing raw log entries with timestamps and services
        """
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
            
        df = self.loader.load_logs_for_time_range(start_time, end_time)
        if df.empty:
            return "No log data found."
            
        # Filter by service if specified
        if service_name:
            df = df[df['cmdb_id'] == service_name]
            
        # Filter by pattern if specified
        if pattern:
            try:
                df = df[df['value'].str.contains(pattern, case=False, na=False, regex=True)]
            except Exception as e:
                return f"Error: Invalid regex pattern: {e}"
                
        if df.empty:
            return "No logs match the specified criteria."
            
        # Sort by timestamp and limit
        df = df.sort_values('datetime').head(limit)
        
        result = [f"Found {len(df)} log entries (showing up to {limit}):"]
        result.append("=" * 80)
        
        for _, row in df.iterrows():
            result.append(f"Time: {row['datetime']}")
            result.append(f"Service: {row['cmdb_id']}")
            result.append(f"Log: {row['value'][:500]}")  # Truncate very long logs
            if len(row['value']) > 500:
                result.append("... (truncated)")
            result.append("-" * 80)
            
        return "\n".join(result)
