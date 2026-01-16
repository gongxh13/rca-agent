import pandas as pd
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from ..utils.schema import (
    COL_TIMESTAMP,
    COL_ENTITY_ID,
    COL_MESSAGE,
    COL_SEVERITY,
    COL_METRIC_NAME,
    COL_VALUE,
    COL_TRACE_ID,
    COL_SPAN_ID,
    COL_PARENT_SPAN_ID,
    COL_DURATION_MS,
    COL_STATUS_CODE,
)


class DataValidationError(Exception):
    pass


class BaseDataLoader(ABC):
    def __init__(self, default_timezone: str = "Asia/Shanghai"):
        self._tz = default_timezone
    
    def get_timezone(self) -> str:
        return self._tz
    
    @abstractmethod
    def load_metrics(self, start_time: str, end_time: str) -> pd.DataFrame:
        raise NotImplementedError()
    
    @abstractmethod
    def load_logs(self, start_time: str, end_time: str) -> pd.DataFrame:
        raise NotImplementedError()
    
    @abstractmethod
    def load_traces(self, start_time: str, end_time: str) -> pd.DataFrame:
        raise NotImplementedError()

    def validate_metrics_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[COL_TIMESTAMP, COL_ENTITY_ID, COL_METRIC_NAME, COL_VALUE])
        required = [COL_TIMESTAMP, COL_ENTITY_ID, COL_METRIC_NAME, COL_VALUE]
        for col in required:
            if col not in df.columns:
                raise DataValidationError(f"metrics missing required column: {col}")
        df = df.copy()
        df[COL_TIMESTAMP] = pd.to_numeric(df[COL_TIMESTAMP], errors="coerce").astype("Int64")
        df[COL_ENTITY_ID] = df[COL_ENTITY_ID].astype(str)
        df[COL_METRIC_NAME] = df[COL_METRIC_NAME].astype(str)
        df[COL_VALUE] = pd.to_numeric(df[COL_VALUE], errors="coerce").astype(float)
        df = df.dropna(subset=[COL_TIMESTAMP, COL_ENTITY_ID, COL_METRIC_NAME, COL_VALUE])
        return df[[COL_TIMESTAMP, COL_ENTITY_ID, COL_METRIC_NAME, COL_VALUE]]

    def validate_logs_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[COL_TIMESTAMP, COL_ENTITY_ID, COL_MESSAGE, COL_SEVERITY])
        required = [COL_TIMESTAMP, COL_ENTITY_ID, COL_MESSAGE]
        for col in required:
            if col not in df.columns:
                raise DataValidationError(f"logs missing required column: {col}")
        df = df.copy()
        df[COL_TIMESTAMP] = pd.to_numeric(df[COL_TIMESTAMP], errors="coerce").astype("Int64")
        df[COL_ENTITY_ID] = df[COL_ENTITY_ID].astype(str)
        df[COL_MESSAGE] = df[COL_MESSAGE].astype(str)
        if COL_SEVERITY not in df.columns:
            df[COL_SEVERITY] = None
        df = df.dropna(subset=[COL_TIMESTAMP, COL_ENTITY_ID, COL_MESSAGE])
        return df[[COL_TIMESTAMP, COL_ENTITY_ID, COL_MESSAGE, COL_SEVERITY]]

    def validate_traces_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[COL_TIMESTAMP, COL_ENTITY_ID, COL_TRACE_ID, COL_SPAN_ID, COL_PARENT_SPAN_ID, COL_DURATION_MS, COL_STATUS_CODE])
        required = [COL_TIMESTAMP, COL_ENTITY_ID, COL_TRACE_ID, COL_SPAN_ID, COL_DURATION_MS]
        for col in required:
            if col not in df.columns:
                raise DataValidationError(f"traces missing required column: {col}")
        df = df.copy()
        df[COL_TIMESTAMP] = pd.to_numeric(df[COL_TIMESTAMP], errors="coerce").astype("Int64")
        df[COL_ENTITY_ID] = df[COL_ENTITY_ID].astype(str)
        df[COL_TRACE_ID] = df[COL_TRACE_ID].astype(str)
        df[COL_SPAN_ID] = df[COL_SPAN_ID].astype(str)
        df[COL_PARENT_SPAN_ID] = df.get(COL_PARENT_SPAN_ID)
        df[COL_DURATION_MS] = pd.to_numeric(df[COL_DURATION_MS], errors="coerce").astype(float)
        if COL_STATUS_CODE not in df.columns:
            df[COL_STATUS_CODE] = None
        df = df.dropna(subset=[COL_TIMESTAMP, COL_ENTITY_ID, COL_TRACE_ID, COL_SPAN_ID, COL_DURATION_MS])
        df = df[df[COL_DURATION_MS] >= 0]
        return df[[COL_TIMESTAMP, COL_ENTITY_ID, COL_TRACE_ID, COL_SPAN_ID, COL_PARENT_SPAN_ID, COL_DURATION_MS, COL_STATUS_CODE]]


    def get_metrics(self, start_time: str, end_time: str) -> pd.DataFrame:
        df = self.load_metrics(start_time, end_time)
        return self.validate_metrics_df(df)

    def get_logs(self, start_time: str, end_time: str) -> pd.DataFrame:
        df = self.load_logs(start_time, end_time)
        return self.validate_logs_df(df)

    def get_traces(self, start_time: str, end_time: str) -> pd.DataFrame:
        df = self.load_traces(start_time, end_time)
        return self.validate_traces_df(df)


class OpenRCADataLoader(BaseDataLoader):
    def __init__(self, dataset_path: str, default_timezone: str = "Asia/Shanghai"):
        super().__init__(default_timezone=default_timezone)
        self.dataset_path = Path(dataset_path)
        self.telemetry_path = self.dataset_path / "telemetry"
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_available_dates(self) -> List[str]:
        if not self.telemetry_path.exists():
            return []
        dates = []
        for item in self.telemetry_path.iterdir():
            if item.is_dir() and item.name.count("_") == 2:
                dates.append(item.name)
        return sorted(dates)

    def _date_to_path_format(self, date_str: str) -> str:
        if "_" in date_str:
            return date_str
        return date_str.replace("-", "_")

    def _get_file_path(self, date: str, data_type: str, filename: str) -> Optional[Path]:
        date_formatted = self._date_to_path_format(date)
        file_path = self.telemetry_path / date_formatted / data_type / filename
        if file_path.exists():
            return file_path
        return None

    def load_metrics(self, start_time: str, end_time: str) -> pd.DataFrame:
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
        dates_to_load = []
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            date_str = current_date.strftime("%Y_%m_%d")
            dates_to_load.append(date_str)
            current_date = pd.Timestamp(current_date) + pd.Timedelta(days=1)
            current_date = current_date.date()
        app_dfs = []
        cont_dfs = []
        for date in dates_to_load:
            app_key = f"metric_app_{date}"
            if app_key in self._cache:
                app_df = self._cache[app_key]
            else:
                app_path = self._get_file_path(date, "metric", "metric_app.csv")
                app_df = pd.read_csv(app_path) if app_path else None
                if app_df is not None:
                    app_df[COL_TIMESTAMP] = (pd.to_numeric(app_df["timestamp"], errors="coerce") * 1000).astype("Int64")
                    self._cache[app_key] = app_df
            if app_df is not None:
                app_dfs.append(app_df)
            cont_key = f"metric_container_{date}"
            if cont_key in self._cache:
                cont_df = self._cache[cont_key]
            else:
                cont_path = self._get_file_path(date, "metric", "metric_container.csv")
                cont_df = pd.read_csv(cont_path) if cont_path else None
                if cont_df is not None:
                    cont_df[COL_TIMESTAMP] = (pd.to_numeric(cont_df["timestamp"], errors="coerce") * 1000).astype("Int64")
                    self._cache[cont_key] = cont_df
            if cont_df is not None:
                cont_dfs.append(cont_df)
        app_df = pd.concat(app_dfs, ignore_index=True) if app_dfs else pd.DataFrame()
        cont_df = pd.concat(cont_dfs, ignore_index=True) if cont_dfs else pd.DataFrame()
        records = []
        if not app_df.empty:
            for _, row in app_df.iterrows():
                svc = row.get("tc")
                ts = row.get(COL_TIMESTAMP)
                for col, kpi in [("mrt", "App_mrt"), ("sr", "App_sr"), ("rr", "App_rr"), ("cnt", "App_cnt")]:
                    if col in app_df.columns:
                        val = row.get(col)
                        records.append(
                            {
                                COL_TIMESTAMP: ts,
                                COL_ENTITY_ID: svc,
                                COL_METRIC_NAME: kpi,
                                COL_VALUE: val,
                            }
                        )
        cont_norm = pd.DataFrame()
        if not cont_df.empty:
            if all(c in cont_df.columns for c in [COL_TIMESTAMP, "cmdb_id", "kpi_name", "value"]):
                cont_norm = cont_df.rename(columns={"cmdb_id": COL_ENTITY_ID, "kpi_name": COL_METRIC_NAME})[
                    [COL_TIMESTAMP, COL_ENTITY_ID, COL_METRIC_NAME, COL_VALUE]
                ]
            else:
                cont_norm = pd.DataFrame()
        app_norm = pd.DataFrame.from_records(records) if records else pd.DataFrame()
        if not app_norm.empty:
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
            app_norm = app_norm[(app_norm[COL_TIMESTAMP] >= start_ms) & (app_norm[COL_TIMESTAMP] <= end_ms)]
        if not cont_norm.empty:
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
            cont_norm = cont_norm[(cont_norm[COL_TIMESTAMP] >= start_ms) & (cont_norm[COL_TIMESTAMP] <= end_ms)]
        if app_norm.empty and cont_norm.empty:
            return pd.DataFrame(columns=[COL_TIMESTAMP, COL_ENTITY_ID, COL_METRIC_NAME, COL_VALUE])

        if app_norm.empty:
            merged = cont_norm
        elif cont_norm.empty:
            merged = app_norm
        else:
            merged = pd.concat([cont_norm, app_norm], ignore_index=True)
        return merged

    def clear_cache(self):
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        return {
            "cached_files": len(self._cache),
            "cache_keys": list(self._cache.keys()),
            "total_rows": sum(len(df) for df in self._cache.values()),
        }

    def load_logs(self, start_time: str, end_time: str) -> pd.DataFrame:
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
        dates_to_load = []
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            date_str = current_date.strftime("%Y_%m_%d")
            dates_to_load.append(date_str)
            current_date = pd.Timestamp(current_date) + pd.Timedelta(days=1)
            current_date = current_date.date()
        dfs = []
        for date in dates_to_load:
            key = f"log_{date}"
            if key in self._cache:
                df = self._cache[key]
            else:
                path = self._get_file_path(date, "log", "log_service.csv")
                df = None
                if path:
                    try:
                        df = pd.read_csv(path)
                        df[COL_TIMESTAMP] = (pd.to_numeric(df["timestamp"], errors="coerce") * 1000).astype("Int64")
                        self._cache[key] = df
                    except Exception:
                        df = None
            if df is not None:
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        if COL_TIMESTAMP in combined.columns:
            combined = combined[(combined[COL_TIMESTAMP] >= start_ms) & (combined[COL_TIMESTAMP] <= end_ms)]
        if not combined.empty and all(c in combined.columns for c in [COL_TIMESTAMP, "cmdb_id", "value"]):
            combined = combined.rename(columns={"cmdb_id": COL_ENTITY_ID, "value": COL_MESSAGE})
        if COL_SEVERITY not in combined.columns:
            combined[COL_SEVERITY] = None
        keep = [COL_TIMESTAMP, COL_ENTITY_ID, COL_MESSAGE, COL_SEVERITY]
        return combined[keep] if all(c in combined.columns for c in keep) else pd.DataFrame()


    def load_traces(self, start_time: str, end_time: str) -> pd.DataFrame:
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
        dates_to_load = []
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            date_str = current_date.strftime("%Y_%m_%d")
            dates_to_load.append(date_str)
            current_date = pd.Timestamp(current_date) + pd.Timedelta(days=1)
            current_date = current_date.date()
        dfs = []
        for date in dates_to_load:
            key = f"trace_{date}"
            if key in self._cache:
                df = self._cache[key]
            else:
                path = self._get_file_path(date, "trace", "trace_span.csv")
                df = pd.read_csv(path) if path else None
                if df is not None:
                    df[COL_TIMESTAMP] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
                    self._cache[key] = df
            if df is not None:
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        if COL_TIMESTAMP in combined.columns:
            combined = combined[(combined[COL_TIMESTAMP] >= start_ms) & (combined[COL_TIMESTAMP] <= end_ms)]
        if not combined.empty:
            rename_map = {"cmdb_id": "entity_id", "parent_id": "parent_span_id", "duration": "duration_ms"}
            for k, v in rename_map.items():
                if k in combined.columns:
                    combined[v] = combined[k]
        
        if COL_STATUS_CODE not in combined.columns:
            combined[COL_STATUS_CODE] = None
            
        keep = [COL_TIMESTAMP, COL_ENTITY_ID, COL_TRACE_ID, COL_SPAN_ID, COL_PARENT_SPAN_ID, COL_DURATION_MS, COL_STATUS_CODE]
        return combined[keep] if all(c in combined.columns for c in keep) else pd.DataFrame()


class DiskFaultDataLoader(BaseDataLoader):
    def __init__(self, dataset_path: str, default_timezone: str = "UTC"):
        super().__init__(default_timezone=default_timezone)
        self.dataset_path = Path(dataset_path)
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_metrics(self, start_time: str, end_time: str) -> pd.DataFrame:
        # Currently we don't have metric files, but we could parse fault_injection_record.csv
        # as a ground truth event stream if needed. For now, return empty.
        return pd.DataFrame(columns=[COL_TIMESTAMP, COL_ENTITY_ID, COL_METRIC_NAME, COL_VALUE])

    def load_traces(self, start_time: str, end_time: str) -> pd.DataFrame:
        return pd.DataFrame(columns=[COL_TIMESTAMP, COL_ENTITY_ID, COL_TRACE_ID, COL_SPAN_ID, COL_PARENT_SPAN_ID, COL_DURATION_MS, COL_STATUS_CODE])

    def load_logs(self, start_time: str, end_time: str) -> pd.DataFrame:
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize(self._tz)
        if end_dt.tzinfo is None:
            end_dt = end_dt.tz_localize(self._tz)

        # Iterate over all daily directories
        dfs = []
        # Assumption: directories are named YYYY-MM-DD
        # We can just iterate all and filter by date, or construct expected paths
        # Constructing paths is more efficient but requires knowing the range
        
        current_date = start_dt.date()
        end_date_val = end_dt.date()
        
        dates_to_load = []
        while current_date <= end_date_val:
            dates_to_load.append(current_date.strftime("%Y-%m-%d"))
            current_date += pd.Timedelta(days=1)
            
        for date_str in dates_to_load:
            day_dir = self.dataset_path / date_str
            if not day_dir.exists():
                continue
            
            for log_file in ["app.log", "kernel.log", "syslog.log"]:
                file_path = day_dir / log_file
                if not file_path.exists():
                    continue
                
                key = f"{date_str}_{log_file}"
                if key in self._cache:
                    dfs.append(self._cache[key])
                    continue
                
                try:
                    # Read file, assuming ISO timestamp at start
                    # We'll read line by line or use pandas if format is consistent
                    # Since lines might be messy, let's read as text and parse
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    
                    data = []
                    entity_id = log_file.split(".")[0]  # app, kernel, syslog
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        # Attempt to split timestamp and message
                        # Format: 2026-01-09T16:55:06.123Z ...
                        # Split by first space
                        parts = line.split(" ", 1)
                        if len(parts) < 2:
                            continue
                        ts_str, msg = parts[0], parts[1]
                        
                        try:
                            ts = pd.to_datetime(ts_str)
                            # Convert to ms int
                            ts_ms = int(ts.timestamp() * 1000)
                            data.append({
                                COL_TIMESTAMP: ts_ms,
                                COL_ENTITY_ID: entity_id,
                                COL_MESSAGE: msg,
                                COL_SEVERITY: "INFO" # Default
                            })
                        except Exception:
                            # If timestamp parse fails, skip or treat as continuation?
                            # For simplicity, skip
                            continue
                    
                    if data:
                        df = pd.DataFrame(data)
                        self._cache[key] = df
                        dfs.append(df)
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

        if not dfs:
            return pd.DataFrame(columns=[COL_TIMESTAMP, COL_ENTITY_ID, COL_MESSAGE, COL_SEVERITY])

        combined = pd.concat(dfs, ignore_index=True)
        
        # Filter by exact time range
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        
        combined = combined[(combined[COL_TIMESTAMP] >= start_ms) & (combined[COL_TIMESTAMP] <= end_ms)]
        
        return combined


_LOADER_REGISTRY: Dict[str, Callable[..., BaseDataLoader]] = {}

def register_data_loader(name: str, constructor: Callable[..., BaseDataLoader]) -> None:
    _LOADER_REGISTRY[name.lower()] = constructor

def create_data_loader(config: Optional[Dict[str, Any]] = None) -> BaseDataLoader:
    cfg = config or {}
    name = str(cfg.get("dataloader", "openrca")).lower()
    dataset_path = cfg.get("dataset_path", "datasets/OpenRCA/Bank")
    if name not in _LOADER_REGISTRY:
        name = "openrca"
    constructor = _LOADER_REGISTRY[name]
    if "default_timezone" in cfg and cfg["default_timezone"]:
        return constructor(dataset_path, default_timezone=cfg["default_timezone"])
    return constructor(dataset_path)

register_data_loader("openrca", lambda dataset_path, default_timezone="Asia/Shanghai": OpenRCADataLoader(dataset_path, default_timezone=default_timezone))
register_data_loader("disk_fault", lambda dataset_path, default_timezone="UTC": DiskFaultDataLoader(dataset_path, default_timezone=default_timezone))
