"""
Data Loader Utility

Utility functions for loading OpenRCA dataset files (metrics, logs, traces).
Handles file discovery, caching, and data preprocessing.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path


class OpenRCADataLoader:
    """
    Loader for OpenRCA dataset files.
    
    Handles loading metrics, logs, and traces from the OpenRCA dataset structure:
    datasets/OpenRCA/{scenario}/telemetry/{date}/{type}/{file}.csv
    """
    
    def __init__(self, dataset_path: str, default_timezone: str = 'Asia/Shanghai'):
        """
        Initialize the data loader.
        
        Args:
            dataset_path: Path to the OpenRCA dataset (e.g., "datasets/OpenRCA/Bank")
        """
        self.dataset_path = Path(dataset_path)
        self.telemetry_path = self.dataset_path / "telemetry"
        self._cache: Dict[str, pd.DataFrame] = {}
        self._tz = default_timezone
        
    def get_available_dates(self) -> List[str]:
        """
        Get list of available dates in the dataset.
        
        Returns:
            List of date strings in YYYY_MM_DD format
        """
        if not self.telemetry_path.exists():
            return []
        
        dates = []
        for item in self.telemetry_path.iterdir():
            if item.is_dir() and item.name.count('_') == 2:
                dates.append(item.name)
        return sorted(dates)
    
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
    
    def load_metric_app(self, date: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load application metrics for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD or YYYY_MM_DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: timestamp, rr, sr, cnt, mrt, tc
        """
        cache_key = f"metric_app_{date}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self._get_file_path(date, "metric", "metric_app.csv")
        if not file_path:
            return None
        
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(self._tz)
        
        if use_cache:
            self._cache[cache_key] = df
        
        return df
    
    def load_metric_container(self, date: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load container/infrastructure metrics for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD or YYYY_MM_DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: timestamp, cmdb_id, kpi_name, value
        """
        cache_key = f"metric_container_{date}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self._get_file_path(date, "metric", "metric_container.csv")
        if not file_path:
            return None
        
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(self._tz)
        
        if use_cache:
            self._cache[cache_key] = df
        
        return df
    
    def load_metrics_for_time_range(
        self,
        start_time: str,
        end_time: str,
        metric_type: str = "app"
    ) -> pd.DataFrame:
        """
        Load metrics for a time range (may span multiple dates).
        
        Args:
            start_time: Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
            end_time: End time in ISO format
            metric_type: Type of metrics ("app" or "container")
            
        Returns:
            Combined DataFrame for the time range
        """
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize(self._tz)
        else:
            start_dt = start_dt.tz_convert(self._tz)
        if end_dt.tzinfo is None:
            end_dt = end_dt.tz_localize(self._tz)
        else:
            end_dt = end_dt.tz_convert(self._tz)
        
        # Get all dates in range
        dates_to_load = []
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            date_str = current_date.strftime("%Y_%m_%d")
            dates_to_load.append(date_str)
            current_date = pd.Timestamp(current_date) + pd.Timedelta(days=1)
            current_date = current_date.date()
        
        # Load and combine data
        dfs = []
        for date in dates_to_load:
            if metric_type == "app":
                df = self.load_metric_app(date)
            else:
                df = self.load_metric_container(date)
            
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=True)
        
        # Filter to exact time range
        combined = combined[
            (combined['datetime'] >= start_dt) & 
            (combined['datetime'] <= end_dt)
        ]
        
        return combined
    
    def load_trace(self, date: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load trace data for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD or YYYY_MM_DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: timestamp, cmdb_id, parent_id, span_id, trace_id, duration
        """
        cache_key = f"trace_{date}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self._get_file_path(date, "trace", "trace_span.csv")
        if not file_path:
            return None
        
        # Traces can be large, so we might want to be careful here
        # For now, load everything as requested
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(self._tz)
        
        if use_cache:
            self._cache[cache_key] = df
        
        return df

    def load_log(self, date: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load log data for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD or YYYY_MM_DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: log_id, timestamp, cmdb_id, log_name, value
        """
        cache_key = f"log_{date}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self._get_file_path(date, "log", "log_service.csv")
        if not file_path:
            return None
        
        # Logs can be very large, so we might want to be careful here
        try:
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(self._tz)
            
            if use_cache:
                self._cache[cache_key] = df
            
            return df
        except Exception as e:
            print(f"Error loading log file for {date}: {e}")
            return None

    def load_logs_for_time_range(
        self,
        start_time: str,
        end_time: str
    ) -> pd.DataFrame:
        """
        Load logs for a time range (may span multiple dates).
        
        Args:
            start_time: Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
            end_time: End time in ISO format
            
        Returns:
            Combined DataFrame for the time range
        """
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize(self._tz)
        else:
            start_dt = start_dt.tz_convert(self._tz)
        if end_dt.tzinfo is None:
            end_dt = end_dt.tz_localize(self._tz)
        else:
            end_dt = end_dt.tz_convert(self._tz)
        
        # Get all dates in range
        dates_to_load = []
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            date_str = current_date.strftime("%Y_%m_%d")
            dates_to_load.append(date_str)
            current_date = pd.Timestamp(current_date) + pd.Timedelta(days=1)
            current_date = current_date.date()
        
        # Load and combine data
        dfs = []
        for date in dates_to_load:
            df = self.load_log(date)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=True)
        
        # Filter to exact time range
        combined = combined[
            (combined['datetime'] >= start_dt) & 
            (combined['datetime'] <= end_dt)
        ]
        
        return combined

    def load_traces_for_time_range(
        self,
        start_time: str,
        end_time: str
    ) -> pd.DataFrame:
        """
        Load traces for a time range (may span multiple dates).
        
        Args:
            start_time: Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
            end_time: End time in ISO format
            
        Returns:
            Combined DataFrame for the time range
        """
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize(self._tz)
        else:
            start_dt = start_dt.tz_convert(self._tz)
        if end_dt.tzinfo is None:
            end_dt = end_dt.tz_localize(self._tz)
        else:
            end_dt = end_dt.tz_convert(self._tz)
        
        # Get all dates in range
        dates_to_load = []
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            date_str = current_date.strftime("%Y_%m_%d")
            dates_to_load.append(date_str)
            current_date = pd.Timestamp(current_date) + pd.Timedelta(days=1)
            current_date = current_date.date()
        
        # Load and combine data
        dfs = []
        for date in dates_to_load:
            df = self.load_trace(date)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=True)
        
        # Filter to exact time range
        combined = combined[
            (combined['datetime'] >= start_dt) & 
            (combined['datetime'] <= end_dt)
        ]
        
        return combined

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_files": len(self._cache),
            "cache_keys": list(self._cache.keys()),
            "total_rows": sum(len(df) for df in self._cache.values())
        }
