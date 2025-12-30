"""
Data Preprocessor for Causal Analysis

This module handles data preprocessing for causal graph construction and root cause inference.
It processes OpenRCA dataset metrics, traces, and logs into formats suitable for causal analysis.
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import json

# Note: We don't use OpenRCADataLoader, we read CSV files directly


class CausalDataPreprocessor:
    """
    Preprocessor for OpenRCA data to prepare it for causal analysis.
    
    Handles:
    - Multi-day data integration
    - Metric aggregation with configurable time granularity
    - Feature engineering (wide table format)
    - Service topology construction from traces
    - Data alignment and missing data handling
    """
    
    # Core metrics to extract
    CORE_METRICS = {
        'cpu': ['OSLinux-CPU_CPU_CPUCpuUtil'],
        'memory': ['OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc'],
        'disk_io': ['DSKRead', 'DSKWrite', 'DSKReadWrite'],
        'disk_space': ['disk', 'space', 'usage'],
        'jvm_cpu': ['_CPULoad'],
        'jvm_oom': ['HeapMemoryMax', 'HeapMemoryUsed'],
        'network': ['NETBandwidthUtil', 'TotalTcpConnNum', 
                   'NetworkRxBytes', 'NetworkTxBytes']
    }
    
    # Metrics to exclude after extraction (used for calculation but not needed in final data)
    EXCLUDED_METRICS = [
        'HeapMemoryMax',      # Used for JVM_Heap_Usage calculation, not needed in final data
        'NETInErr',           # Network error metrics excluded from causal analysis
        'NETInErrPrc',        # Network error percentage metrics excluded
        'NETOutErr',          # Network output error metrics excluded
        'NETOutErrPrcc',      # Network output error percentage metrics excluded (note: Prcc typo)
    ]
    
    # Candidate components (from OpenRCA)
    CANDIDATE_COMPONENTS = [
        'apache01', 'apache02',
        'Tomcat01', 'Tomcat02', 'Tomcat03', 'Tomcat04',
        'Mysql01', 'Mysql02',
        'Redis01', 'Redis02',
        'MG01', 'MG02',
        'IG01', 'IG02'
    ]
    
    def __init__(
        self,
        dataset_path: str,
        time_granularity: str = '5min',
        compact_mode: bool = True
    ):
        """
        Initialize the data preprocessor.
        
        Args:
            dataset_path: Path to OpenRCA dataset (e.g., "datasets/OpenRCA/Bank")
            time_granularity: Time granularity for aggregation (e.g., '1min', '5min', '10min')
            compact_mode: If True, only keep key metrics based on fault types in record.csv,
                   and remove similar/redundant metrics. Default: True
        """
        self.dataset_path = Path(dataset_path)
        self.telemetry_path = self.dataset_path / "telemetry"
        self.time_granularity = time_granularity
        self.compact_mode = compact_mode
        
        # Load fault types from record.csv if compact_mode is enabled
        self.fault_types = None
        if self.compact_mode:
            self.fault_types = self._load_fault_types()
        
    def _load_fault_types(self) -> Dict[str, List[str]]:
        """
        Load fault types from record.csv and map them to key metrics.
        
        Returns:
            Dictionary mapping fault types to key metric patterns
        """
        record_path = self.dataset_path / "record.csv"
        if not record_path.exists():
            print(f"Warning: record.csv not found at {record_path}, using default fault types")
            return self._get_default_fault_types()
        
        try:
            record_df = pd.read_csv(record_path)
            if 'reason' not in record_df.columns:
                print("Warning: 'reason' column not found in record.csv, using default fault types")
                return self._get_default_fault_types()
            
            # Get unique fault types
            unique_faults = record_df['reason'].unique().tolist()
            print(f"Found {len(unique_faults)} unique fault types in record.csv: {unique_faults}")
            
            # Map fault types to key metrics
            fault_to_metrics = {}
            for fault in unique_faults:
                if 'high CPU usage' in fault.lower():
                    fault_to_metrics[fault] = ['OSLinux-CPU_CPU_CPUCpuUtil']
                elif 'high memory usage' in fault.lower():
                    fault_to_metrics[fault] = ['OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc']
                elif 'high disk I/O' in fault.lower() or 'disk I/O read' in fault.lower():
                    # For disk I/O, prefer read metrics (most common in faults)
                    fault_to_metrics[fault] = ['DSKRead']
                elif 'high disk space' in fault.lower():
                    fault_to_metrics[fault] = ['disk', 'space', 'usage']
                elif 'high JVM CPU load' in fault.lower():
                    fault_to_metrics[fault] = ['_CPULoad']
                elif 'JVM Out of Memory' in fault or 'OOM' in fault:
                    fault_to_metrics[fault] = ['JVM_Heap_Usage']  # Use calculated metric
                elif 'network latency' in fault.lower() or 'network packet loss' in fault.lower():
                    # For network issues, prefer bandwidth utilization
                    fault_to_metrics[fault] = ['NETBandwidthUtil']
                else:
                    # Unknown fault type, use default
                    fault_to_metrics[fault] = []
            
            return fault_to_metrics
            
        except Exception as e:
            print(f"Error loading record.csv: {e}, using default fault types")
            return self._get_default_fault_types()
    
    def _get_default_fault_types(self) -> Dict[str, List[str]]:
        """
        Get default fault types based on common OpenRCA faults.
        
        Returns:
            Dictionary mapping fault types to key metric patterns
        """
        return {
            'high CPU usage': ['OSLinux-CPU_CPU_CPUCpuUtil'],
            'high memory usage': ['OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc'],
            'high disk I/O read usage': ['DSKRead'],
            'high disk space usage': ['disk', 'space', 'usage'],
            'high JVM CPU load': ['_CPULoad'],
            'JVM Out of Memory (OOM) Heap': ['JVM_Heap_Usage'],
            'network latency': ['NETBandwidthUtil'],
            'network packet loss': ['NETBandwidthUtil']
        }
    
    def _get_compact_metric_patterns(self) -> List[str]:
        """
        Get key metric patterns for compact mode based on fault types.
        
        Returns:
            List of key metric patterns to keep
        """
        if not self.fault_types:
            # Fallback to default patterns
            patterns = []
            for metrics in self._get_default_fault_types().values():
                patterns.extend(metrics)
            return list(set(patterns))
        
        # Collect all key metric patterns from fault types
        key_patterns = []
        for metrics in self.fault_types.values():
            key_patterns.extend(metrics)
        
        # Remove duplicates and return
        return list(set(key_patterns))
    
    def _remove_similar_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove similar/redundant metrics in compact mode.
        Keep only the most representative metric for each metric type.
        
        Args:
            df: DataFrame with metrics
            
        Returns:
            DataFrame with similar metrics removed
        """
        if df.empty or not self.compact_mode:
            return df
        
        # Group metrics by type and keep only the most representative one
        # Strategy: For each metric type (CPU, memory, disk, network, JVM),
        # keep only the most common or most representative metric
        
        df_filtered = df.copy()
        
        # For disk I/O metrics, prefer DSKRead over DSKWrite and DSKReadWrite
        # (based on record.csv, most disk I/O faults are read-related)
        disk_io_metrics = df_filtered[df_filtered['kpi_name'].str.contains('DSK', na=False)]
        if not disk_io_metrics.empty:
            # Keep only DSKRead metrics, remove DSKWrite and DSKReadWrite
            write_metrics = df_filtered['kpi_name'].str.contains('DSKWrite', na=False)
            readwrite_metrics = df_filtered['kpi_name'].str.contains('DSKReadWrite', na=False)
            df_filtered = df_filtered[~(write_metrics | readwrite_metrics)]
        
        # For network metrics, prefer NETBandwidthUtil over other network metrics
        # (based on record.csv, network faults are mostly latency/packet loss)
        network_metrics = df_filtered[df_filtered['kpi_name'].str.contains('NET', na=False) | 
                                       df_filtered['kpi_name'].str.contains('Network', na=False)]
        if not network_metrics.empty:
            # Keep NETBandwidthUtil and TotalTcpConnNum, remove others
            # TotalTcpConnNum is also important for network analysis
            keep_network = (
                df_filtered['kpi_name'].str.contains('NETBandwidthUtil', na=False) |
                df_filtered['kpi_name'].str.contains('TotalTcpConnNum', na=False)
            )
            remove_network = (
                (df_filtered['kpi_name'].str.contains('NET', na=False) |
                 df_filtered['kpi_name'].str.contains('Network', na=False)) &
                ~keep_network
            )
            df_filtered = df_filtered[~remove_network]
        
        # For JVM metrics, if we have JVM_Heap_Usage (calculated), remove HeapMemoryUsed
        # (JVM_Heap_Usage is more informative)
        if 'JVM_Heap_Usage' in df_filtered['kpi_name'].values:
            df_filtered = df_filtered[
                ~df_filtered['kpi_name'].str.contains('HeapMemoryUsed', na=False)
            ]
        
        return df_filtered
    
    def _date_to_path_format(self, date_str: str) -> str:
        """
        Convert ISO date string to path format.
        
        Args:
            date_str: Date in ISO format (YYYY-MM-DD) or path format (YYYY_MM_DD)
            
        Returns:
            Date in path format (YYYY_MM_DD)
        """
        if '_' in date_str:
            return date_str
        return date_str.replace('-', '_')
    
    def _get_file_path(self, date: str, data_type: str, filename: str) -> Optional[Path]:
        """
        Get the full path to a data file.
        
        Args:
            date: Date string (YYYY-MM-DD or YYYY_MM_DD)
            data_type: Type of data (metric, log, trace)
            filename: Name of the file
            
        Returns:
            Path to the file, or None if it doesn't exist
        """
        date_formatted = self._date_to_path_format(date)
        file_path = self.telemetry_path / date_formatted / data_type / filename
        
        if file_path.exists():
            return file_path
        return None
    
    def _load_metric_file(
        self,
        date: str,
        metric_type: str,
        chunksize: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load metric file for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD or YYYY_MM_DD)
            metric_type: Type of metrics ('app' or 'container')
            chunksize: If provided, read file in chunks to save memory
            
        Returns:
            DataFrame with datetime column in UTC (no timezone conversion)
        """
        filename = f"metric_{metric_type}.csv"
        file_path = self._get_file_path(date, "metric", filename)
        
        if not file_path:
            return None
        
        try:
            # Use chunk reading for large files to save memory
            if chunksize:
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunksize):
                    chunk['datetime'] = pd.to_datetime(chunk['timestamp'], unit='s', utc=True)
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _load_trace_file(
        self, 
        date: str,
        usecols: Optional[List[str]] = None,
        chunksize: Optional[int] = None,
        sample_ratio: Optional[float] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load trace file for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD or YYYY_MM_DD)
            usecols: Only read these columns to save memory (default: ['span_id', 'cmdb_id', 'parent_id'])
            chunksize: If provided, read file in chunks to save memory
            sample_ratio: If provided (0-1), randomly sample this ratio of rows to reduce memory
            
        Returns:
            DataFrame with datetime column in UTC (no timezone conversion)
        """
        file_path = self._get_file_path(date, "trace", "trace_span.csv")
        
        if not file_path:
            return None
        
        try:
            # Default behavior: include timestamp for datetime conversion
            # Only skip timestamp when explicitly requested via usecols
            if usecols is None:
                usecols = ['span_id', 'cmdb_id', 'parent_id', 'timestamp']
            
            # Use chunk reading for large files to save memory
            if chunksize:
                chunks = []
                for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunksize):
                    # Convert timestamp if it's in the columns
                    if 'timestamp' in chunk.columns and 'datetime' not in chunk.columns:
                        chunk['datetime'] = pd.to_datetime(chunk['timestamp'], unit='ms', utc=True)
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path, usecols=usecols)
                # Convert timestamp if it's in the columns
                if 'timestamp' in df.columns and 'datetime' not in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # Sample data if requested to reduce memory
            if sample_ratio and 0 < sample_ratio < 1:
                df = df.sample(frac=sample_ratio, random_state=42)
            
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_all_data(
        self,
        start_date: str,
        end_date: str,
        metric_type: str = 'container',
        chunksize: Optional[int] = 100000
    ) -> pd.DataFrame:
        """
        Load all data for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            metric_type: Type of metrics ('container' or 'app')
            chunksize: Chunk size for reading large files (default: 100000)
            
        Returns:
            Combined DataFrame for the date range
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get all dates in range
        dates = []
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            date_str = current_date.strftime("%Y_%m_%d")
            dates.append(date_str)
            current_date += timedelta(days=1)
        
        # Load and combine data incrementally to save memory
        combined = None
        for i, date in enumerate(dates):
            print(f"  Loading {date} ({i+1}/{len(dates)})...", end=' ', flush=True)
            df = self._load_metric_file(date, metric_type, chunksize=chunksize)
            
            if df is not None and not df.empty:
                if combined is None:
                    combined = df
                else:
                    # Incremental concat to save memory
                    combined = pd.concat([combined, df], ignore_index=True)
                    # Delete intermediate DataFrame to free memory
                    del df
            print("Done")
        
        if combined is None:
            return pd.DataFrame()
        
        return combined
    
    def extract_core_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract core metrics from container metrics DataFrame.
        
        In compact mode, only keeps key metrics based on fault types in record.csv
        and removes similar/redundant metrics.
        
        Args:
            df: Container metrics DataFrame
            
        Returns:
            DataFrame with only core metrics
        """
        if df.empty:
            return df
        
        # Filter by candidate components
        df = df[df['cmdb_id'].isin(self.CANDIDATE_COMPONENTS)]
        
        if self.compact_mode:
            # In compact mode, use fault-based metric selection
            df_filtered = self._extract_metrics_compact_mode(df)
        else:
            # Original mode: use all core metrics
            df_filtered = self._extract_metrics_normal_mode(df)
        
        # Special handling for JVM OOM: calculate heap usage
        if not df_filtered.empty:
            df_filtered = self._process_jvm_oom(df_filtered)
        
        # Remove excluded metrics (used for calculation but not needed in final data)
        if not df_filtered.empty:
            df_filtered = self._remove_excluded_metrics(df_filtered)
        
        # In compact mode, remove similar metrics
        if self.compact_mode and not df_filtered.empty:
            df_filtered = self._remove_similar_metrics(df_filtered)
        
        return df_filtered
    
    def _extract_metrics_compact_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract metrics in compact mode based on fault types.
        
        Args:
            df: Container metrics DataFrame
            
        Returns:
            DataFrame with only key metrics based on fault types
        """
        # Get key metric patterns from fault types
        key_patterns = self._get_compact_metric_patterns()
        
        if not key_patterns:
            # Fallback to normal mode if no patterns found
            return self._extract_metrics_normal_mode(df)
        
        # Build metric mask based on key patterns
        metric_mask = False
        for pattern in key_patterns:
            if pattern == 'OSLinux-CPU_CPU_CPUCpuUtil':
                metric_mask |= df['kpi_name'].str.contains(pattern, na=False)
            elif pattern == 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc':
                metric_mask |= df['kpi_name'].str.contains(pattern, na=False)
            elif pattern == 'DSKRead':
                metric_mask |= df['kpi_name'].str.endswith(pattern, na=False)
            elif pattern in ['disk', 'space', 'usage']:
                metric_mask |= (
                    df['kpi_name'].str.contains('disk', case=False, na=False) &
                    (df['kpi_name'].str.contains('space', case=False, na=False) |
                     df['kpi_name'].str.contains('usage', case=False, na=False))
                )
            elif pattern == '_CPULoad':
                metric_mask |= (
                    df['kpi_name'].str.contains('JVM', na=False) &
                    df['kpi_name'].str.contains(pattern, na=False)
                )
            elif pattern == 'JVM_Heap_Usage':
                # This will be calculated later, but we need HeapMemoryMax and HeapMemoryUsed
                metric_mask |= (
                    df['kpi_name'].str.contains('HeapMemoryMax', na=False) |
                    df['kpi_name'].str.contains('HeapMemoryUsed', na=False)
                )
            elif pattern == 'NETBandwidthUtil':
                metric_mask |= (
                    df['kpi_name'].str.contains(pattern, na=False) |
                    df['kpi_name'].str.contains('TotalTcpConnNum', na=False)
                )
        
        return df[metric_mask].copy()
    
    def _extract_metrics_normal_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract metrics in normal mode (all core metrics).
        
        Args:
            df: Container metrics DataFrame
            
        Returns:
            DataFrame with all core metrics
        """
        # Filter by core metrics
        metric_mask = False
        for metric_type, patterns in self.CORE_METRICS.items():
            for pattern in patterns:
                if metric_type == 'disk_io':
                    metric_mask |= df['kpi_name'].str.endswith(pattern, na=False)
                elif metric_type == 'disk_space':
                    metric_mask |= (
                        df['kpi_name'].str.contains('disk', case=False, na=False) &
                        (df['kpi_name'].str.contains('space', case=False, na=False) |
                         df['kpi_name'].str.contains('usage', case=False, na=False))
                    )
                elif metric_type == 'jvm_cpu':
                    metric_mask |= (
                        df['kpi_name'].str.contains('JVM', na=False) &
                        df['kpi_name'].str.contains(pattern, na=False)
                    )
                elif metric_type == 'jvm_oom':
                    metric_mask |= df['kpi_name'].str.contains(pattern, na=False)
                elif metric_type == 'network':
                    metric_mask |= (
                        df['kpi_name'].str.contains(pattern, na=False) |
                        df['kpi_name'].str.endswith(pattern, na=False)
                    )
                else:
                    metric_mask |= df['kpi_name'].str.contains(pattern, na=False)
        
        return df[metric_mask].copy()
    
    def _process_jvm_oom(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process JVM OOM metrics by calculating heap usage.
        
        Args:
            df: DataFrame with JVM metrics
            
        Returns:
            DataFrame with added JVM heap usage metrics
        """
        heap_max_df = df[df['kpi_name'].str.contains('HeapMemoryMax', na=False)].copy()
        heap_used_df = df[df['kpi_name'].str.contains('HeapMemoryUsed', na=False)].copy()
        
        if heap_max_df.empty or heap_used_df.empty:
            return df
        
        # Calculate heap usage for each component
        jvm_oom_data = []
        components = set(heap_max_df['cmdb_id'].unique()).intersection(
            set(heap_used_df['cmdb_id'].unique())
        )
        
        for comp in components:
            max_data = heap_max_df[heap_max_df['cmdb_id'] == comp].sort_values('datetime')
            used_data = heap_used_df[heap_used_df['cmdb_id'] == comp].sort_values('datetime')
            
            # Align by time (1 minute window)
            max_data['time_key'] = max_data['datetime'].dt.floor('1min')
            used_data['time_key'] = used_data['datetime'].dt.floor('1min')
            
            merged = pd.merge(
                max_data[['time_key', 'value']].rename(columns={'value': 'HeapMemoryMax'}),
                used_data[['time_key', 'value']].rename(columns={'value': 'HeapMemoryUsed'}),
                on='time_key',
                how='inner'
            )
            
            if not merged.empty:
                merged['HeapUsage'] = merged['HeapMemoryUsed'] / merged['HeapMemoryMax']
                merged = merged[merged['HeapMemoryMax'] > 0]
                
                if not merged.empty:
                    for _, row in merged.iterrows():
                        jvm_oom_data.append({
                            'timestamp': row['time_key'].timestamp(),
                            'cmdb_id': comp,
                            'kpi_name': 'JVM_Heap_Usage',
                            'value': row['HeapUsage'],
                            'datetime': row['time_key']
                        })
        
        if jvm_oom_data:
            jvm_oom_df = pd.DataFrame(jvm_oom_data)
            # Combine with original data
            df = pd.concat([df, jvm_oom_df], ignore_index=True)
        
        return df
    
    def _remove_excluded_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove metrics that are used for calculation but not needed in final data.
        
        This includes:
        - HeapMemoryMax: Used to calculate JVM_Heap_Usage, but the original metric is not needed
        - Network error metrics: NETInErr, NETInErrPrc, NETOutErr, NETOutErrPrcc
        
        Args:
            df: DataFrame with metrics
            
        Returns:
            DataFrame with excluded metrics removed
        """
        if df.empty:
            return df
        
        # Build exclusion mask
        exclude_mask = False
        for pattern in self.EXCLUDED_METRICS:
            exclude_mask |= df['kpi_name'].str.contains(pattern, na=False)
        
        # Remove excluded metrics
        df_filtered = df[~exclude_mask].copy()
        
        return df_filtered
    
    def aggregate_metrics(
        self,
        df: pd.DataFrame,
        metric_type: str = 'container'
    ) -> pd.DataFrame:
        """
        Aggregate metrics by time window and component/service.
        
        Args:
            df: Raw metrics DataFrame
            metric_type: Type of metrics ('container' or 'app')
            
        Returns:
            Aggregated DataFrame
        """
        if df.empty:
            return df
        
        df = df.copy()
        df.set_index('datetime', inplace=True)
        
        if metric_type == 'container':
            # Group by component, metric, and time window
            grouped = df.groupby(['cmdb_id', 'kpi_name']).resample(self.time_granularity)
            aggregated = grouped['value'].mean().reset_index()
        else:  # app metrics
            # Group by service and time window
            grouped = df.groupby('tc').resample(self.time_granularity)
            aggregated = grouped[['rr', 'sr', 'cnt', 'mrt']].mean().reset_index()
        
        return aggregated
    
    def create_wide_table(
        self,
        container_df: pd.DataFrame,
        app_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create wide table format for causal analysis.
        
        Each row is a time point, each column is a metric.
        Column naming: {component}_{metric_type} or {service}_{metric}
        
        Args:
            container_df: Aggregated container metrics DataFrame
            app_df: Optional aggregated app metrics DataFrame
            
        Returns:
            Wide table DataFrame
        """
        wide_tables = []
        
        # Process container metrics
        if not container_df.empty:
            # Create feature names: {component}_{metric_type}
            container_df['feature_name'] = (
                container_df['cmdb_id'].astype(str) + '_' + 
                container_df['kpi_name'].astype(str)
            )
            
            # Pivot to wide format
            container_wide = container_df.pivot_table(
                index='datetime',
                columns='feature_name',
                values='value',
                aggfunc='mean'
            )
            wide_tables.append(container_wide)
        
        # Process app metrics
        if app_df is not None and not app_df.empty:
            # Create feature names: {service}_{metric}
            for metric in ['rr', 'sr', 'cnt', 'mrt']:
                if metric in app_df.columns:
                    app_metric = app_df.pivot_table(
                        index='datetime',
                        columns='tc',
                        values=metric,
                        aggfunc='mean'
                    )
                    app_metric.columns = [f"{col}_{metric}" for col in app_metric.columns]
                    wide_tables.append(app_metric)
        
        # Merge all wide tables
        if not wide_tables:
            return pd.DataFrame()
        
        # Align by datetime index
        result = wide_tables[0]
        for table in wide_tables[1:]:
            result = result.join(table, how='outer')
        
        # Fill missing values (forward fill then backward fill)
        result = result.ffill().bfill()
        
        return result
    
    def build_service_topology(
        self, 
        start_date: str, 
        end_date: str,
        chunksize: Optional[int] = 100000,
        sample_ratio: Optional[float] = 0.05,
        max_records: Optional[int] = None
    ) -> nx.DiGraph:
        """
        Build service dependency topology from trace data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            chunksize: Chunk size for reading large trace files (default: 100000)
            sample_ratio: Sample ratio for trace data to reduce memory (default: 0.05 = 5%)
                         For topology building, we don't need all traces, sampling is sufficient
            max_records: Maximum number of trace records to process (default: None = no limit)
                        If set, will randomly sample this many records from loaded data
            
        Returns:
            NetworkX DiGraph representing service dependencies
        """
        # Load trace data for date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        dates = []
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            date_str = current_date.strftime("%Y_%m_%d")
            dates.append(date_str)
            current_date += timedelta(days=1)
        
        # Load and combine trace data incrementally
        # Only load columns we need: span_id, cmdb_id, parent_id (no timestamp needed for topology)
        all_traces = None
        for i, date in enumerate(dates):
            print(f"  Loading traces for {date} ({i+1}/{len(dates)})...", end=' ', flush=True)
            trace_df = self._load_trace_file(
                date, 
                usecols=['span_id', 'cmdb_id', 'parent_id'],  # No timestamp needed for topology
                chunksize=chunksize,
                sample_ratio=sample_ratio
            )
            if trace_df is not None and not trace_df.empty:
                # Only keep the columns we need
                trace_data = trace_df[['span_id', 'cmdb_id', 'parent_id']].copy()
                if all_traces is None:
                    all_traces = trace_data
                else:
                    # Incremental concat
                    all_traces = pd.concat([all_traces, trace_data], ignore_index=True)
                # Delete intermediate DataFrame to free memory
                del trace_df, trace_data
            print("Done")
        
        if all_traces is None or all_traces.empty:
            return nx.DiGraph()
        
        # Further limit records if max_records is set
        if max_records and len(all_traces) > max_records:
            print(f"  Sampling {max_records:,} from {len(all_traces):,} records...", end=' ', flush=True)
            all_traces = all_traces.sample(n=max_records, random_state=42)
            print("Done")
        
        print(f"  Processing {len(all_traces):,} trace records...", end=' ', flush=True)
        
        # Build dependency graph using dictionary mapping (much more efficient than merge)
        # Strategy: Create span_id -> cmdb_id mapping, then iterate through parent_id
        
        # Step 1: Create span_id -> cmdb_id mapping dictionary (O(n) time, O(n) space)
        print("  Creating span mapping...", end=' ', flush=True)
        # Use dict.fromkeys + update for better memory handling for large datasets
        # But for simplicity and correctness, use direct zip (pandas handles memory efficiently)
        span_to_service = dict(zip(all_traces['span_id'], all_traces['cmdb_id']))
        print("Done")
        
        # Step 2: Build edge counts using dictionary (O(n) time, O(m) space where m = unique edges)
        print("  Building edges...", end=' ', flush=True)
        edge_counts = {}  # (parent_service, child_service) -> count
        
        # Use itertuples() which is much faster than iterrows() for large datasets
        # itertuples() returns namedtuples which are faster to access
        total_records = len(all_traces)
        for idx, row in enumerate(all_traces.itertuples(index=False)):
            parent_id = row.parent_id
            child_service = row.cmdb_id
            
            # Skip if parent_id is None or empty (root spans)
            if pd.isna(parent_id) or parent_id == '':
                continue
            
            # Look up parent service from mapping
            parent_service = span_to_service.get(parent_id)
            
            # Skip if parent service not found or self-loop
            if parent_service is None or parent_service == child_service:
                continue
            
            # Increment edge count
            edge_key = (parent_service, child_service)
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
            
            # Progress indicator for very large datasets
            if (idx + 1) % 1000000 == 0:
                print(f"\n    Processed {idx + 1:,} / {total_records:,} records...", end=' ', flush=True)
        
        print("Done")
        
        # Step 3: Create graph from edge counts
        print("  Creating graph...", end=' ', flush=True)
        G = nx.DiGraph()
        
        # Add edges with weights
        for (parent, child), weight in edge_counts.items():
            G.add_edge(parent, child, weight=weight)
        
        # Clean up
        del span_to_service, edge_counts, all_traces
        
        print("Done")
        return G
    
    def prepare_causal_data(
        self,
        start_date: str,
        end_date: str,
        include_app_metrics: bool = True,
        chunksize: Optional[int] = 100000,
        trace_sample_ratio: Optional[float] = 0.1
    ) -> Dict[str, Any]:
        """
        Prepare all data for causal analysis.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            include_app_metrics: Whether to include application metrics
            chunksize: Chunk size for reading large files (default: 100000)
            trace_sample_ratio: Sample ratio for trace data (default: 0.1 = 10%)
            
        Returns:
            Dictionary containing:
            - 'wide_table': Wide table DataFrame
            - 'service_topology': Service topology graph
            - 'data_alignment_report': Data alignment report
            - 'data_quality_report': Data quality report
        """
        print(f"Loading data from {start_date} to {end_date}...")
        
        # Load container metrics
        print("Loading container metrics...")
        container_df = self.load_all_data(start_date, end_date, metric_type='container', chunksize=chunksize)
        print(f"Loaded {len(container_df)} container metric records")
        
        # Extract core metrics
        mode_str = "compact mode (fault-based)" if self.compact_mode else "normal mode"
        print(f"Extracting core metrics ({mode_str})...")
        container_df = self.extract_core_metrics(container_df)
        print(f"After filtering core metrics: {len(container_df)} records")
        
        # Aggregate container metrics
        print("Aggregating container metrics...")
        container_agg = self.aggregate_metrics(container_df, metric_type='container')
        print(f"Aggregated to {len(container_agg)} time points")
        
        # Free memory: delete original container_df after aggregation
        del container_df
        
        # Load app metrics (optional)
        app_agg = None
        app_df = None
        if include_app_metrics:
            print("Loading application metrics...")
            app_df = self.load_all_data(start_date, end_date, metric_type='app', chunksize=chunksize)
            if not app_df.empty:
                print(f"Loaded {len(app_df)} app metric records")
                app_agg = self.aggregate_metrics(app_df, metric_type='app')
                print(f"Aggregated to {len(app_agg)} time points")
        
        # Create wide table
        print("Creating wide table...")
        wide_table = self.create_wide_table(container_agg, app_agg)
        print(f"Wide table shape: {wide_table.shape}")
        print(f"Features: {list(wide_table.columns)[:10]}...")  # Show first 10 features
        
        # Free memory: delete aggregated dataframes after creating wide table
        del container_agg
        if app_agg is not None:
            del app_agg
        
        # Build service topology
        print("Building service topology...")
        service_topology = self.build_service_topology(
            start_date, 
            end_date,
            chunksize=chunksize,
            sample_ratio=trace_sample_ratio,
            max_records=5000000  # Limit to 5M records max for very large datasets
        )
        print(f"Service topology: {service_topology.number_of_nodes()} nodes, "
              f"{service_topology.number_of_edges()} edges")
        
        # Generate reports (need to reload container_df for alignment report)
        # But we can use wide_table info instead to avoid reloading
        print("Generating reports...")
        data_alignment_report = self._generate_alignment_report_from_wide_table(wide_table)
        data_quality_report = self._generate_quality_report(wide_table)
        
        return {
            'wide_table': wide_table,
            'service_topology': service_topology,
            'data_alignment_report': data_alignment_report,
            'data_quality_report': data_quality_report
        }
    
    def _generate_alignment_report(
        self,
        container_df: pd.DataFrame,
        app_df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Generate data alignment report."""
        report = {
            'container_components': sorted(container_df['cmdb_id'].unique().tolist()) if not container_df.empty else [],
            'container_metrics_count': len(container_df) if not container_df.empty else 0,
        }
        
        if app_df is not None and not app_df.empty:
            report['app_services'] = sorted(app_df['tc'].unique().tolist())
            report['app_metrics_count'] = len(app_df)
        else:
            report['app_services'] = []
            report['app_metrics_count'] = 0
        
        return report
    
    def _generate_alignment_report_from_wide_table(
        self,
        wide_table: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate data alignment report from wide table to avoid reloading data."""
        if wide_table.empty:
            return {
                'container_components': [],
                'container_metrics_count': 0,
                'app_services': [],
                'app_metrics_count': 0
            }
        
        # Extract container components from column names
        # Format: {component}_{metric_name}
        container_components = set()
        app_services = set()
        
        for col in wide_table.columns:
            # Container metrics: {component}_{metric}
            # App metrics: {service}_{metric} (rr, sr, cnt, mrt)
            parts = col.split('_')
            if len(parts) >= 2:
                # Check if it's an app metric (ends with rr, sr, cnt, mrt)
                if parts[-1] in ['rr', 'sr', 'cnt', 'mrt']:
                    # App metric: {service}_{metric}
                    service = '_'.join(parts[:-1])
                    app_services.add(service)
                else:
                    # Container metric: {component}_{metric}
                    # Try to identify component (first part or first few parts)
                    # This is heuristic, but should work for most cases
                    component = parts[0]
                    if component in self.CANDIDATE_COMPONENTS:
                        container_components.add(component)
        
        report = {
            'container_components': sorted(list(container_components)),
            'container_metrics_count': len([c for c in wide_table.columns if not any(c.endswith(f'_{m}') for m in ['rr', 'sr', 'cnt', 'mrt'])]),
            'app_services': sorted(list(app_services)),
            'app_metrics_count': len([c for c in wide_table.columns if any(c.endswith(f'_{m}') for m in ['rr', 'sr', 'cnt', 'mrt'])])
        }
        
        return report
    
    def _generate_quality_report(self, wide_table: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report."""
        if wide_table.empty:
            return {'status': 'empty', 'missing_rate': 1.0}
        
        missing_rate = wide_table.isna().sum().sum() / (wide_table.shape[0] * wide_table.shape[1])
        
        return {
            'status': 'ok',
            'shape': list(wide_table.shape),
            'missing_rate': float(missing_rate),
            'feature_count': wide_table.shape[1],
            'time_points': wide_table.shape[0],
            'date_range': {
                'start': str(wide_table.index.min()),
                'end': str(wide_table.index.max())
            }
        }
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        datetime_as_timestamp: bool = True
    ):
        """
        Save preprocessing results to files.
        
        Args:
            results: Results from prepare_causal_data
            output_dir: Output directory
            datetime_as_timestamp: If True, save datetime as UTC timestamp (seconds).
                                  If False, save as ISO format string.
                                  Default: True (more efficient and avoids timezone issues)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save wide table
        if not results['wide_table'].empty:
            wide_table = results['wide_table'].copy()
            
            # Convert datetime index to timestamp or ISO string
            if datetime_as_timestamp:
                # Convert to UTC timestamp (seconds)
                # Handle timezone-aware datetime index
                if hasattr(wide_table.index, 'tz') and wide_table.index.tz is not None:
                    # Convert to UTC first, then to timestamp
                    wide_table.index = wide_table.index.tz_convert('UTC')
                # Convert to Unix timestamp (seconds since epoch)
                wide_table.index = wide_table.index.astype('int64') // 10**9
                wide_table.index.name = 'timestamp_utc'
            else:
                # Keep as ISO format string
                if hasattr(wide_table.index, 'strftime'):
                    wide_table.index = wide_table.index.strftime('%Y-%m-%d %H:%M:%S%z')
                else:
                    wide_table.index = wide_table.index.astype(str)
                wide_table.index.name = 'datetime'
            
            wide_table.to_csv(output_path / 'all_data.csv')
            print(f"Saved wide table to {output_path / 'all_data.csv'}")
            print(f"  Datetime format: {'UTC timestamp (seconds)' if datetime_as_timestamp else 'ISO string'}")
        
        # Save service topology
        if results['service_topology'].number_of_nodes() > 0:
            nx.write_graphml(results['service_topology'], output_path / 'service_topology.graphml')
            print(f"Saved service topology to {output_path / 'service_topology.graphml'}")
        
        # Save reports
        with open(output_path / 'data_alignment_report.json', 'w') as f:
            json.dump(results['data_alignment_report'], f, indent=2, default=str)
        
        with open(output_path / 'data_quality_report.json', 'w') as f:
            json.dump(results['data_quality_report'], f, indent=2, default=str)
        
        print(f"Saved reports to {output_path}")

