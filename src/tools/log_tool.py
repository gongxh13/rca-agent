"""
Log Analysis Tool

Provides high-level semantic operations for analyzing log data in the context
of root cause analysis.
"""

from typing import Any, Dict, List, Optional

from .base import BaseRCATool
import pandas as pd
import pickle
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from .data_loader import BaseDataLoader, create_data_loader
from ..utils.time_utils import to_iso_with_tz
from ..utils.schema import (
    COL_TIMESTAMP,
    COL_ENTITY_ID,
    COL_MESSAGE,
    COL_SEVERITY,
)


class LogAnalysisTool(BaseRCATool):
    """
    Tool for analyzing log data with high-level semantic operations.
    
    This tool is designed to work with log data from various sources and provide
    insights without overwhelming the agent with raw log entries. It focuses on
    pattern detection, error analysis, and temporal correlations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the log analysis tool.
        
        Args:
            config: Configuration dictionary that may include:
                   - log_source_path: Path to log files or database
                   - log_format: Format of the logs (json, text, etc.)
                   - index_fields: Fields to index for faster querying
        """
        super().__init__(config)
        self.data_loader: Optional[BaseDataLoader] = None
        
    def initialize(self) -> None:
        super().initialize()
        self.data_loader = create_data_loader(self.config)
    
    def _get_log_df(self, start_time: str, end_time: str) -> pd.DataFrame:
        return self.data_loader.get_logs(start_time, end_time) if self.data_loader else pd.DataFrame(columns=[COL_TIMESTAMP,COL_ENTITY_ID,COL_MESSAGE,COL_SEVERITY])
    
    def get_tools(self) -> List[Any]:
        """Get list of LangChain tools for log analysis."""
        return [
            self.wrap(self.get_log_summary),
            self.wrap(self.query_logs),
            self.wrap(self.extract_log_templates_drain3),
        ]
    
    def get_log_summary(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        entity_id: Optional[str] = None
    ) -> str:
        """Get a high-level summary of log activity.
        
        Provides aggregated statistics about log volume, error rates, and activity patterns.
        
        Args:
            start_time: Start of time range in ISO format
            end_time: End of time range in ISO format
            entity_id: Optional filter by entity identifier
            
        Returns:
            A formatted summary including total entries, error counts, warning counts,
            and most active entities
        """
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
        df = self._get_log_df(start_time, end_time)
        if df.empty:
            return "No log data found."
        if entity_id:
            df = df[df[COL_ENTITY_ID] == entity_id]
        total_logs = len(df)
        entities = df[COL_ENTITY_ID].nunique()
        error_count = df[COL_MESSAGE].str.contains("error|exception|fail", case=False, na=False).sum()
        warning_count = df[COL_MESSAGE].str.contains("warn", case=False, na=False).sum()
        top_entities = df[COL_ENTITY_ID].value_counts().head(5)
        result = [
            "Log Summary:",
            f"- Total Entries: {total_logs}",
            f"- Unique Entities: {entities}",
            f"- Error Count: {error_count} ({(error_count/total_logs)*100:.1f}%)",
            f"- Warning Count: {warning_count}",
            "\nTop Active Entities:",
        ]
        for ent, count in top_entities.items():
            result.append(f"- {ent}: {count} entries")
        return "\n".join(result)
    
    def query_logs(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        entity_id: Optional[str] = None,
        pattern: Optional[str] = None,
        limit: int = 20
    ) -> str:
        """Query and view raw log entries.
        
        This tool allows viewing actual log content for detailed investigation.
        Use this after identifying patterns or anomalies to examine specific log entries.
        
        Args:
            start_time: Start of time range in ISO format
            end_time: End of time range in ISO format
            entity_id: Optional filter by entity identifier
            pattern: Optional regex pattern to match in log content (case-insensitive)
            limit: Maximum number of log entries to return (default: 20)
            
        Returns:
            Formatted string containing raw log entries with timestamps, entities, and content
        """
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
        df = self._get_log_df(start_time, end_time)
        if df.empty:
            return "No log data found."
        if entity_id:
            df = df[df[COL_ENTITY_ID] == entity_id]
        if pattern:
            try:
                df = df[df[COL_MESSAGE].str.contains(pattern, case=False, na=False, regex=True)]
            except Exception as e:
                return f"Error: Invalid regex pattern: {e}"
        if df.empty:
            return "No logs match the specified criteria."
        df = df.sort_values(COL_TIMESTAMP).head(limit)
        result = [f"Found {len(df)} log entries (showing up to {limit}):", "=" * 80]
        for _, row in df.iterrows():
            ts_val = row[COL_TIMESTAMP]
            tz = self.data_loader.get_timezone()
            result.append(f"Time: {to_iso_with_tz(ts_val, tz)}")
            result.append(f"Entity: {row[COL_ENTITY_ID]}")
            msg = str(row[COL_MESSAGE])
            result.append(f"Log: {msg[:500]}")
            if len(msg) > 500:
                result.append("... (truncated)")
            result.append("-" * 80)
        return "\n".join(result)
    
    def extract_log_templates_drain3(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        entity_id: Optional[str] = None,
        top_n: int = 50,
        min_count: int = 2,
        config_path: Optional[str] = None,
        include_params: bool = False,
        model_path: Optional[str] = None
    ) -> str:
        """Extract log templates with Drain3 within a time window and count frequencies.
        
        Performs template-level pattern mining using Drain3:
        - Online mining: incrementally clusters the current window’s logs into templates, no pretraining required
        - Pretrained matching: load a pretrained Drain3 miner (model_path) and re-count matches within the window
        - Optional parameter examples: provide one parameter example per template to illustrate variable parts
        
        Usage recommendations:
        - For single-window analysis, online mining is sufficient
        - For cross-window template stability or faster matching, pretrain and provide model_path
        - For stronger normalization, provide config_path to enable masking rules
        
        Args:
            start_time: Start time (ISO, YYYY-MM-DDTHH:MM:SS)
            end_time: End time (ISO)
            entity_id: Filter by entity identifier; None means all entities
            top_n: Maximum number of templates to return
            min_count: Minimum frequency threshold to include a template
            config_path: Path to Drain3 INI configuration
            include_params: Whether to return one parameter example per template
            model_path: Path to pretrained Drain3 miner pickle
        
        Returns:
            Text summary: templates with counts and optional parameter examples
        """
        if not start_time or not end_time:
            return "Error: start_time and end_time are required"
        df = self._get_log_df(start_time, end_time)
        if df.empty:
            return "No log data found."
        if entity_id:
            df = df[df[COL_ENTITY_ID] == entity_id]
            if df.empty:
                return f"No logs found for entity {entity_id}."
        messages = df[COL_MESSAGE].dropna().astype(str).tolist()
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
        else:
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
            class _C:
                __slots__ = ("cluster_id", "size", "_tpl")
                def __init__(self, cid, size, tpl):
                    self.cluster_id = cid
                    self.size = size
                    self._tpl = tpl
                def get_template(self):
                    return self._tpl
            clusters = [_C(cid, cnt, cluster_template[cid]) for cid, cnt in window_counts.items()]
            clusters.sort(key=lambda c: c.size, reverse=True)
        result_lines = []
        header = "Log Templates (Drain3)"
        if entity_id:
            header += f" - Entity: {entity_id}"
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
    
    def cleanup(self) -> None:
        if self.data_loader:
            self.data_loader = None
        super().cleanup()
