"""
Local Log Analysis Tool

Implementation of LogAnalysisTool for OpenRCA dataset.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import pickle

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from .log_tool import LogAnalysisTool
from .data_loader import OpenRCADataLoader
from src.utils.time_utils import to_iso_shanghai

class LocalLogAnalysisTool(LogAnalysisTool):
    """
    Local implementation of LogAnalysisTool using OpenRCA dataset files.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.loader = None
        self._pre_cfg: Dict[str, Any] = self.config.get("preprocess", {})
        
    def initialize(self) -> None:
        """Initialize the data loader."""
        dataset_path = self.config.get("dataset_path", "datasets/OpenRCA/Bank")
        default_tz = self.config.get("default_timezone", "Asia/Shanghai")
        self.loader = OpenRCADataLoader(dataset_path, default_timezone=default_tz)
    
    def get_tools(self) -> List[Any]:
        return super().get_tools()
        
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
                f"- {a['service']} at {to_iso_shanghai(a['time'])}: {a['count']} logs/min "
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
    
    def extract_log_templates_drain3(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        service_name: Optional[str] = None,
        top_n: int = 1000,
        min_count: int = 2,
        config_path: Optional[str] = None,
        include_params: bool = False,
        model_path: Optional[str] = None
    ) -> str:
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
        
        df = self.loader.load_logs_for_time_range(start_time, end_time)
        if df.empty:
            return "No log data found."
        
        if service_name:
            df = df[df["cmdb_id"] == service_name]
            if df.empty:
                return f"No logs found for service {service_name}."
        
        messages = df["value"].dropna().astype(str).tolist()
        messages = self._preprocess_messages(messages)
        if not messages:
            return "No log messages available for template mining."
        
        miner = None
        if model_path:
            try:
                with open(model_path, "rb") as f:
                    miner = pickle.load(f)
            except Exception as e:
                return f"Error: failed to load Drain3 model from {model_path}: {e}"
        
        if miner is None:
            config = TemplateMinerConfig()
            if config_path:
                try:
                    config.load(config_path)
                except Exception as e:
                    return f"Error: failed to load Drain3 config from {config_path}: {e}"
            miner = TemplateMiner(config=config)
            for msg in messages:
                miner.add_log_message(msg.rstrip())
            
            clusters = list(miner.drain.clusters)
            if not clusters:
                return "No templates discovered."
            clusters.sort(key=lambda c: c.size, reverse=True)
            use_window_counts = False
        else:
            # 预训练模型：按当前窗口重计数
            window_counts: Dict[int, int] = {}
            cluster_template: Dict[int, str] = {}
            for msg in messages:
                cluster = miner.match(msg)
                if cluster is None:
                    continue
                cid = cluster.cluster_id
                window_counts[cid] = window_counts.get(cid, 0) + 1
                if cid not in cluster_template:
                    cluster_template[cid] = cluster.get_template()
            if not window_counts:
                return "No templates matched for current window."
            # 构造临时“集群”视图用于统一输出
            class _C:
                __slots__ = ("cluster_id", "size", "_tpl")
                def __init__(self, cid, size, tpl):
                    self.cluster_id = cid
                    self.size = size
                    self._tpl = tpl
                def get_template(self):
                    return self._tpl
            clusters = [ _C(cid, cnt, cluster_template[cid]) for cid, cnt in window_counts.items() ]
            clusters.sort(key=lambda c: c.size, reverse=True)
            use_window_counts = True
        
        result_lines = []
        header = "Log Templates (Drain3)"
        if service_name:
            header += f" - Service: {service_name}"
        header += f"\nTime Range: {start_time} ~ {end_time}"
        result_lines.append(header)
        result_lines.append("=" * 80)
        
        shown = 0
        for c in clusters:
            if c.size < min_count:
                continue
            template = c.get_template()
            line = f"[cluster #{c.cluster_id}] count={c.size} template={template}"
            result_lines.append(line)
            if include_params:
                try:
                    sample_params = None
                    for msg in messages[:500]:
                        cluster = miner.match(msg)
                        if cluster and cluster.cluster_id == c.cluster_id:
                            sample_params = miner.get_parameter_list(template, msg)
                            break
                    if sample_params:
                        result_lines.append(f"  params_example={sample_params}")
                except Exception:
                    pass
            
            shown += 1
            if shown >= top_n:
                break
        
        if shown == 0:
            return f"No templates with count >= {min_count}."
        
        return "\n".join(result_lines)
    
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
            result.append(f"Time: {to_iso_shanghai(row['datetime'])}")
            result.append(f"Service: {row['cmdb_id']}")
            result.append(f"Log: {row['value'][:500]}")  # Truncate very long logs
            if len(row['value']) > 500:
                result.append("... (truncated)")
            result.append("-" * 80)
            
        return "\n".join(result)
    
    def train_log_templates_drain3(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        service_name: Optional[str] = None,
        config_path: Optional[str] = None,
        output_model_path: Optional[str] = None,
        output_templates_csv: Optional[str] = None,
        max_messages: Optional[int] = None
    ) -> str:
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig
        except Exception as e:
            return f"Error: drain3 is not installed or failed to import: {e}"
        df = self.loader.load_logs_for_time_range(start_time, end_time)
        if df.empty:
            return "No log data found."
        if service_name:
            df = df[df["cmdb_id"] == service_name]
            if df.empty:
                return f"No logs found for service {service_name}."
        messages = df["value"].dropna().astype(str).tolist()
        if max_messages is not None:
            messages = messages[:max_messages]
        messages = self._preprocess_messages(messages)
        if not messages:
            return "No log messages available for training."
        config = TemplateMinerConfig()
        if config_path:
            try:
                config.load(config_path)
            except Exception as e:
                return f"Error: failed to load Drain3 config from {config_path}: {e}"
        miner = TemplateMiner(config=config)
        for msg in messages:
            miner.add_log_message(msg.rstrip())
        clusters = list(miner.drain.clusters)
        clusters.sort(key=lambda c: c.size, reverse=True)
        import os
        if output_model_path:
            try:
                os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
                with open(output_model_path, "wb") as f:
                    pickle.dump(miner, f)
            except Exception as e:
                return f"Error: failed to save model to {output_model_path}: {e}"
        if output_templates_csv:
            try:
                os.makedirs(os.path.dirname(output_templates_csv), exist_ok=True)
            except Exception:
                pass
            try:
                import pandas as _pd
                data = {
                    "cluster_id": [c.cluster_id for c in clusters],
                    "template": [c.get_template() for c in clusters],
                    "count": [c.size for c in clusters],
                }
                df_templates = _pd.DataFrame(data)
                df_templates.to_csv(output_templates_csv, index=False)
            except Exception as e:
                return f"Error: failed to save templates csv to {output_templates_csv}: {e}"
        lines = []
        hdr = "Drain3 offline training completed"
        if service_name:
            hdr += f" - Service: {service_name}"
        hdr += f"\nTime Range: {start_time} ~ {end_time}"
        lines.append(hdr)
        if output_model_path:
            lines.append(f"Model saved: {output_model_path}")
        if output_templates_csv:
            lines.append(f"Templates CSV: {output_templates_csv}")
        lines.append("=" * 80)
        shown = 0
        for c in clusters[:50]:
            lines.append(f"[cluster #{c.cluster_id}] count={c.size} template={c.get_template()}")
            shown += 1
            if shown >= 20:
                break
        return "\n".join(lines)
    
    def batch_train_log_templates_drain3(
        self,
        date_list: Optional[List[str]] = None,
        by_service: bool = False,
        service_whitelist: Optional[List[str]] = None,
        output_dir: str = "artifacts/drain3/batch",
        config_path: Optional[str] = None,
        service_limit: Optional[int] = None,
        max_messages_per_service: Optional[int] = None,
        max_messages_per_date: Optional[int] = None
    ) -> str:
        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig
        except Exception as e:
            return f"Error: drain3 is not installed or failed to import: {e}"
        import os
        import pandas as _pd
        os.makedirs(output_dir, exist_ok=True)
        if date_list is None:
            date_list = self.loader.get_available_dates()
        if not date_list:
            return "No dates available for training."
        config = TemplateMinerConfig()
        if config_path:
            try:
                config.load(config_path)
            except Exception as e:
                return f"Error: failed to load Drain3 config from {config_path}: {e}"
        summaries = []
        for date in date_list:
            df = self.loader.load_log(date)
            if df is None or df.empty:
                summaries.append(f"{date}: no log data")
                continue
            messages_date = df["value"].dropna().astype(str).tolist()
            if max_messages_per_date is not None:
                messages_date = messages_date[:max_messages_per_date]
            messages_date = self._preprocess_messages(messages_date)
            miner_date = TemplateMiner(config=config)
            for msg in messages_date:
                miner_date.add_log_message(msg.rstrip())
            clusters_date = list(miner_date.drain.clusters)
            clusters_date.sort(key=lambda c: c.size, reverse=True)
            date_out_dir = os.path.join(output_dir, date)
            os.makedirs(date_out_dir, exist_ok=True)
            try:
                with open(os.path.join(date_out_dir, "model.pkl"), "wb") as f:
                    pickle.dump(miner_date, f)
            except Exception as e:
                return f"Error: failed to save model for {date}: {e}"
            try:
                data = {
                    "cluster_id": [c.cluster_id for c in clusters_date],
                    "template": [c.get_template() for c in clusters_date],
                    "count": [c.size for c in clusters_date],
                }
                _pd.DataFrame(data).to_csv(os.path.join(date_out_dir, "templates.csv"), index=False)
            except Exception as e:
                return f"Error: failed to save templates csv for {date}: {e}"
            summaries.append(f"{date}: date-level clusters={len(clusters_date)}")
            if by_service:
                svc_counts = df["cmdb_id"].value_counts()
                services = svc_counts.index.tolist()
                if service_whitelist:
                    services = [s for s in services if s in service_whitelist]
                if service_limit is not None:
                    services = services[:service_limit]
                for svc in services:
                    svc_msgs = df[df["cmdb_id"] == svc]["value"].dropna().astype(str).tolist()
                    if max_messages_per_service is not None:
                        svc_msgs = svc_msgs[:max_messages_per_service]
                    if not svc_msgs:
                        continue
                    svc_msgs = self._preprocess_messages(svc_msgs)
                    miner_svc = TemplateMiner(config=config)
                    for msg in svc_msgs:
                        miner_svc.add_log_message(msg.rstrip())
                    clusters_svc = list(miner_svc.drain.clusters)
                    clusters_svc.sort(key=lambda c: c.size, reverse=True)
                    svc_out_dir = os.path.join(date_out_dir, "services", svc)
                    os.makedirs(svc_out_dir, exist_ok=True)
                    try:
                        with open(os.path.join(svc_out_dir, "model.pkl"), "wb") as f:
                            pickle.dump(miner_svc, f)
                    except Exception as e:
                        return f"Error: failed to save model for {date}/{svc}: {e}"
                    try:
                        data = {
                            "cluster_id": [c.cluster_id for c in clusters_svc],
                            "template": [c.get_template() for c in clusters_svc],
                            "count": [c.size for c in clusters_svc],
                        }
                        _pd.DataFrame(data).to_csv(os.path.join(svc_out_dir, "templates.csv"), index=False)
                    except Exception as e:
                        return f"Error: failed to save templates csv for {date}/{svc}: {e}"
                    summaries.append(f"{date}/{svc}: clusters={len(clusters_svc)}")
        return "\n".join(summaries)
    
    def _preprocess_messages(self, messages: List[str]) -> List[str]:
        cfg = self._pre_cfg or {}
        if not cfg.get("enabled", False):
            return messages
        max_len = cfg.get("max_length")
        collapse_stack = cfg.get("collapse_stack", False)
        regex_replacements = cfg.get("regex_replacements") or []
        out: List[str] = []
        for m in messages:
            s = m
            if collapse_stack:
                s = re.sub(r'(?:\s+at [\w\.$]+\([^)]*\))+', ' <STACK_TRACE>', s)
            if regex_replacements:
                for rr in regex_replacements:
                    try:
                        pat = rr.get("pattern")
                        repl = rr.get("repl", "")
                        if pat:
                            s = re.sub(pat, repl, s)
                    except Exception:
                        pass
            if isinstance(max_len, int) and max_len > 0 and len(s) > max_len:
                s = s[:max_len] + " <TRUNC>"
            out.append(s)
        return out
    
    def train_log_templates_drain3_cumulative(
        self,
        date_list: Optional[List[str]] = None,
        by_service: bool = False,
        service_whitelist: Optional[List[str]] = None,
        output_dir: str = "artifacts/drain3/cumulative",
        config_path: Optional[str] = None,
        max_messages_total: Optional[int] = None,
        max_messages_per_service_total: Optional[int] = None,
    ) -> str:
        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig
        except Exception as e:
            return f"Error: drain3 is not installed or failed to import: {e}"
        import os
        import pandas as _pd
        os.makedirs(output_dir, exist_ok=True)
        if date_list is None:
            date_list = self.loader.get_available_dates()
        if not date_list:
            return "No dates available for training."
        config = TemplateMinerConfig()
        if config_path:
            try:
                config.load(config_path)
            except Exception as e:
                return f"Error: failed to load Drain3 config from {config_path}: {e}"
        summaries = []
        if not by_service:
            miner = TemplateMiner(config=config)
            msgs: List[str] = []
            for date in date_list:
                df = self.loader.load_log(date)
                if df is None or df.empty:
                    continue
                part = df["value"].dropna().astype(str).tolist()
                msgs.extend(part)
                if max_messages_total is not None and len(msgs) >= max_messages_total:
                    msgs = msgs[:max_messages_total]
                    break
            msgs = self._preprocess_messages(msgs)
            for msg in msgs:
                miner.add_log_message(msg.rstrip())
            clusters = list(miner.drain.clusters)
            clusters.sort(key=lambda c: c.size, reverse=True)
            try:
                with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
                    pickle.dump(miner, f)
            except Exception as e:
                return f"Error: failed to save cumulative model: {e}"
            try:
                data = {
                    "cluster_id": [c.cluster_id for c in clusters],
                    "template": [c.get_template() for c in clusters],
                    "count": [c.size for c in clusters],
                }
                _pd.DataFrame(data).to_csv(os.path.join(output_dir, "templates.csv"), index=False)
            except Exception as e:
                return f"Error: failed to save cumulative templates csv: {e}"
            summaries.append(f"cumulative: clusters={len(clusters)}")
        else:
            svc_msgs_map: Dict[str, List[str]] = {}
            for date in date_list:
                df = self.loader.load_log(date)
                if df is None or df.empty:
                    continue
                for svc, group in df.groupby("cmdb_id"):
                    if service_whitelist and svc not in service_whitelist:
                        continue
                    msgs = group["value"].dropna().astype(str).tolist()
                    buf = svc_msgs_map.get(svc, [])
                    buf.extend(msgs)
                    svc_msgs_map[svc] = buf
            for svc, msgs in svc_msgs_map.items():
                if max_messages_per_service_total is not None and len(msgs) > max_messages_per_service_total:
                    msgs = msgs[:max_messages_per_service_total]
                miner_svc = TemplateMiner(config=config)
                for msg in msgs:
                    miner_svc.add_log_message(msg.rstrip())
                clusters_svc = list(miner_svc.drain.clusters)
                clusters_svc.sort(key=lambda c: c.size, reverse=True)
                svc_out_dir = os.path.join(output_dir, "services", svc)
                os.makedirs(svc_out_dir, exist_ok=True)
                try:
                    with open(os.path.join(svc_out_dir, "model.pkl"), "wb") as f:
                        pickle.dump(miner_svc, f)
                except Exception as e:
                    return f"Error: failed to save cumulative service model for {svc}: {e}"
                try:
                    data = {
                        "cluster_id": [c.cluster_id for c in clusters_svc],
                        "template": [c.get_template() for c in clusters_svc],
                        "count": [c.size for c in clusters_svc],
                    }
                    _pd.DataFrame(data).to_csv(os.path.join(svc_out_dir, "templates.csv"), index=False)
                except Exception as e:
                    return f"Error: failed to save cumulative service templates csv for {svc}: {e}"
                summaries.append(f"cumulative/{svc}: clusters={len(clusters_svc)}")
        return "\n".join(summaries)
