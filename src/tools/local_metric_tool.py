"""
Local Metric Analysis Tool

Concrete implementation of MetricAnalysisTool for the OpenRCA dataset.
Uses local CSV files via OpenRCADataLoader to provide metric analysis.
"""

from typing import Any, Dict, List, Optional, Literal
import pandas as pd
import numpy as np
from datetime import datetime
import json

from .metric_tool import MetricAnalysisTool
from .data_loader import OpenRCADataLoader
from src.utils.time_utils import to_iso_shanghai

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False


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

    # Infrastructure Metrics Tools
    
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
        """
        Detect anomalies in core metrics using ruptures or Z-score methods.
        
        This is a robust tool that focuses on core metrics (CPU, memory, disk, network, JVM)
        for candidate components and identifies anomalies with fault start times.
        
        Args:
            start_time: Start time in ISO format
            end_time: End time in ISO format
            method: Detection method - "ruptures", "zscore", or "both" (default: "both")
            component_id: Optional component ID to filter
            sensitivity: Z-score threshold for anomaly detection (default: 3.0)
            top: Maximum number of anomalies to return (default: 10)
            ruptures_algorithm: Algorithm for ruptures - "pelt", "binseg", "dynp", "window" (default: "pelt")
            ruptures_model: Model for ruptures - "rbf", "l1", "l2", "linear", "normal", "ar", "rank" (default: "rbf")
            pen: Penalty parameter for ruptures (default: 5.0)
            z_threshold: Z-score threshold (default: None, uses sensitivity if None)
            min_data_points_ruptures: Minimum data points for ruptures (default: 10)
            min_data_points_zscore: Minimum data points for z-score (default: 5)
            min_consecutive: Minimum consecutive anomalies for z-score (default: 3)
        
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
        self._check_loader()
        
        # 候选组件列表
        CANDIDATE_COMPONENTS = [
            "apache01", "apache02",
            "Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04",
            "Mysql01", "Mysql02",
            "Redis01", "Redis02",
            "MG01", "MG02",
            "IG01", "IG02"
        ]
        
        # 规范化时间（处理时区）
        def normalize_time(time_input: str) -> pd.Timestamp:
            try:
                dt = pd.to_datetime(time_input)
                if dt.tzinfo is None:
                    dt = dt.tz_localize('Asia/Shanghai')
                else:
                    dt = dt.tz_convert('Asia/Shanghai')
            except Exception:
                dt = pd.to_datetime(time_input, errors='coerce')
                if pd.isna(dt):
                    raise ValueError(f"无法解析时间格式: {time_input}")
                if dt.tzinfo is None:
                    dt = dt.tz_localize('Asia/Shanghai')
                else:
                    dt = dt.tz_convert('Asia/Shanghai')
            return dt
        
        start_dt = normalize_time(start_time)
        end_dt = normalize_time(end_time)
        start_time_str = start_dt.strftime('%Y-%m-%dT%H:%M:%S')
        end_time_str = end_dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        # 加载数据
        df = self.data_loader.load_metrics_for_time_range(
            start_time=start_time_str,
            end_time=end_time_str,
            metric_type="container"
        )
        
        if df.empty:
            return json.dumps([], ensure_ascii=False, indent=2)
            
        # 筛选候选组件
        if component_id:
            components = [component_id]
        else:
            components = CANDIDATE_COMPONENTS
        
        df = df[df['cmdb_id'].isin(components)].copy()
        
        if df.empty:
            return json.dumps([], ensure_ascii=False, indent=2)
        
        # 筛选核心指标
        def is_core_metric(kpi_name: str) -> tuple[bool, Optional[str]]:
            import re
            kpi_lower = kpi_name.lower()
            if kpi_name == 'OSLinux-CPU_CPU_CPUCpuUtil':
                return True, 'cpu'
            if kpi_name == 'OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc':
                return True, 'memory'
            # 磁盘I/O：使用正则表达式匹配
            # 磁盘读：".*DSKRead$"，磁盘写：".*DSKWrite$"，磁盘读写：".*DSKReadWrite$"
            if re.match(r'.*DSKRead$', kpi_name) or re.match(r'.*DSKWrite$', kpi_name) or re.match(r'.*DSKReadWrite$', kpi_name):
                return True, 'disk_io'
            # 磁盘空间：包含disk和space或usage关键词
            if 'disk' in kpi_lower and ('space' in kpi_lower or 'usage' in kpi_lower):
                return True, 'disk_space'
            # 网络指标：关键网络性能指标
            # 网络带宽利用率
            if 'NETBandwidthUtil' in kpi_name:
                return True, 'network_bandwidth'
            # 网络错误：输入/输出错误
            if re.match(r'.*NETInErr.*$', kpi_name) or re.match(r'.*NETOutErr.*$', kpi_name):
                return True, 'network_error'
            # TCP连接数：总连接数和异常状态连接
            if 'TotalTcpConnNum' in kpi_name or 'TCP-CLOSE-WAIT' in kpi_name or 'TCP-FIN-WAIT' in kpi_name:
                return True, 'network_connection'
            # 容器网络流量：接收和发送字节数
            if re.match(r'.*NetworkRxBytes$', kpi_name) or re.match(r'.*NetworkTxBytes$', kpi_name):
                return True, 'network_container'
            # JVM CPU Load：只匹配JVM相关的CPULoad，不匹配系统CPU Load
            # 系统CPU Load是 OSLinux-CPU_CPU_CPULoad，不应该被识别为核心指标
            if 'JVM' in kpi_name and '_CPULoad' in kpi_name:
                return True, 'jvm_cpu'
            if 'HeapMemoryMax' in kpi_name or 'HeapMemoryUsed' in kpi_name:
                return True, 'jvm_oom'
            return False, None
        
        # 处理JVM OOM指标
        def process_jvm_oom(df: pd.DataFrame) -> pd.DataFrame:
            heap_max_df = df[df['kpi_name'].str.contains('HeapMemoryMax', na=False)].copy()
            heap_used_df = df[df['kpi_name'].str.contains('HeapMemoryUsed', na=False)].copy()
            
            if heap_max_df.empty or heap_used_df.empty:
                return pd.DataFrame()
            
            jvm_oom_data = []
            components_with_heap = set(heap_max_df['cmdb_id'].unique()).intersection(
                set(heap_used_df['cmdb_id'].unique())
            )
            
            for comp in components_with_heap:
                max_data = heap_max_df[heap_max_df['cmdb_id'] == comp].sort_values('datetime')
                used_data = heap_used_df[heap_used_df['cmdb_id'] == comp].sort_values('datetime')
                
                max_data['time_key'] = max_data['datetime'].dt.floor('min')
                used_data['time_key'] = used_data['datetime'].dt.floor('min')
                
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
                return pd.DataFrame(jvm_oom_data)
            return pd.DataFrame()
        
        # 筛选核心指标
        core_metrics = []
        for kpi in df['kpi_name'].unique():
            is_core, metric_type = is_core_metric(kpi)
            if is_core and metric_type != 'jvm_oom':
                core_metrics.append(df[df['kpi_name'] == kpi])
        
        jvm_oom_df = process_jvm_oom(df)
        if not jvm_oom_df.empty:
            core_metrics.append(jvm_oom_df)
        
        if not core_metrics:
            return json.dumps([], ensure_ascii=False, indent=2)
        
        df_core = pd.concat(core_metrics, ignore_index=True)
        
        # 处理参数
        ruptures_algorithm = ruptures_algorithm.lower()
        ruptures_model = ruptures_model.lower()
        if z_threshold is None:
            z_threshold = sensitivity
        
        # 验证算法和模型参数
        valid_algorithms = ['pelt', 'binseg', 'dynp', 'window']
        valid_models = ['rbf', 'l1', 'l2', 'linear', 'normal', 'ar', 'rank', 'mahalanobis']
        
        # 默认使用 Pelt + rbf 的原因：
        # - Pelt: 快速且精确，线性时间复杂度，适合大多数场景，能找到全局最优解
        # - rbf: 径向基函数核，能捕捉非线性模式，对复杂的时间序列数据表现良好
        # 
        # 其他算法选择建议：
        # - binseg: 当数据量很大且需要快速检测时使用（可能不是全局最优）
        # - dynp: 当需要保证全局最优且数据量不大时使用（计算成本高）
        # - window: 当需要在线检测或实时监控时使用
        #
        # 其他模型选择建议：
        # - l1: 当数据包含异常值时使用（对异常值更鲁棒）
        # - l2: 标准最小二乘，适合线性趋势
        # - linear: 明确知道数据是线性趋势时使用
        # - normal: 数据符合高斯分布时使用
        # - ar: 时间序列有自相关依赖时使用
        # - rank: 非参数方法，对分布假设不敏感
        
        if ruptures_algorithm not in valid_algorithms:
            ruptures_algorithm = 'pelt'  # 默认使用 pelt
        if ruptures_model not in valid_models:
            ruptures_model = 'rbf'  # 默认使用 rbf
        
        def get_threshold(kpi_name: str) -> float:
            if 'CPU' in kpi_name or 'CPULoad' in kpi_name:
                return 20.0
            elif 'MEM' in kpi_name or 'Memory' in kpi_name or 'Heap_Usage' in kpi_name:
                return 30.0
            elif 'DSK' in kpi_name or 'disk' in kpi_name.lower():
                return 50.0
            elif 'NET' in kpi_name or 'Network' in kpi_name:
                # 网络带宽利用率：80%以上需要关注
                if 'BandwidthUtil' in kpi_name:
                    return 80.0
                # 网络错误：任何错误都需要关注
                elif 'Err' in kpi_name:
                    return 1.0
                # TCP连接数：变化超过50%需要关注
                elif 'TcpConnNum' in kpi_name or 'TCP-' in kpi_name:
                    return 50.0
                # 网络流量：变化超过50%需要关注
                else:
                    return 50.0
            else:
                return 30.0
        
        def get_absolute_threshold(kpi_name: str) -> Optional[float]:
            """获取绝对阈值，用于检测持续高值（即使没有变化点）"""
            if 'MEM' in kpi_name or 'Memory' in kpi_name:
                # 内存使用率超过85%认为是异常
                return 85.0
            elif 'JVM' in kpi_name and 'CPULoad' in kpi_name:
                # JVM CPU Load超过20%认为是异常（JVM CPU Load通常较低，20%已经很高）
                return 20.0
            elif 'CPU' in kpi_name or 'CPULoad' in kpi_name:
                # 系统CPU使用率超过80%认为是异常
                return 80.0
            elif 'Heap_Usage' in kpi_name:
                # JVM堆使用率超过90%认为是异常
                return 0.9
            elif 'BandwidthUtil' in kpi_name:
                # 网络带宽利用率超过85%认为是异常
                return 85.0
            return None
        
        def get_baseline_params(kpi_name: str) -> tuple[float, float]:
            """
            获取基线参数：最小基线阈值和参考值
            用于处理小基线值的情况，避免除以接近0的值导致误报
            
            Returns:
                (min_baseline_threshold, reference_value)
            """
            kpi_lower = kpi_name.lower()
            # CPULoad类型的指标（绝对值很小，通常在0-1之间）
            if 'cpuload' in kpi_lower:
                return (0.1, 1.0)  # 基线阈值0.1，参考值1.0（100% CPU Load）
            # CPU使用率（百分比，通常在0-100之间）
            elif 'cpu' in kpi_lower and 'util' in kpi_lower:
                return (10.0, 100.0)  # 基线阈值10%，参考值100%
            # 内存使用率（百分比，通常在0-100之间）
            elif 'mem' in kpi_lower or 'memory' in kpi_lower:
                return (10.0, 100.0)  # 基线阈值10%，参考值100%
            # JVM堆使用率（比例，通常在0-1之间）
            elif 'heap' in kpi_lower:
                return (0.1, 1.0)  # 基线阈值0.1，参考值1.0
            # 网络带宽利用率（百分比）
            elif 'bandwidth' in kpi_lower or 'util' in kpi_lower:
                return (10.0, 100.0)  # 基线阈值10%，参考值100%
            # 默认值：对于其他指标，使用较小的阈值
            else:
                return (0.1, 100.0)  # 默认基线阈值0.1，参考值100%
        
        def calculate_severity(deviation_pct: float, max_value: float) -> str:
            if deviation_pct > 100:
                severity = "严重"
            elif deviation_pct > 50:
                severity = "显著"
            else:
                severity = "中等"
            return f"{severity}（最大值：{max_value:.1f}，偏离：{deviation_pct:.1f}%）"
        
        # 使用ruptures检测
        def detect_with_ruptures(component: str, kpi_name: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
            if not RUPTURES_AVAILABLE or len(data) < min_data_points_ruptures:
                return []
            
            data = data.sort_values('datetime').reset_index(drop=True)
            values = data['value'].values.astype(float)
            # 直接从DataFrame获取datetime列，保留时区信息
            datetime_series = data['datetime']
            
            anomalies = []
            try:
                signal = values.reshape(-1, 1)
                
                # 根据选择的算法创建检测器
                if ruptures_algorithm == 'pelt':
                    algo = rpt.Pelt(model=ruptures_model).fit(signal)
                    change_points = algo.predict(pen=pen)
                elif ruptures_algorithm == 'binseg':
                    algo = rpt.Binseg(model=ruptures_model).fit(signal)
                    # Binseg可以使用penalty或n_bkps参数
                    # 优先使用penalty参数，如果penalty无效则使用n_bkps
                    max_n_bkps = max(2, min(10, len(values) // 10))
                    try:
                        change_points = algo.predict(pen=pen)
                    except (TypeError, ValueError):
                        # 如果penalty参数不支持，使用n_bkps
                        change_points = algo.predict(n_bkps=max_n_bkps)
                elif ruptures_algorithm == 'dynp':
                    algo = rpt.Dynp(model=ruptures_model).fit(signal)
                    # Dynp必须指定最大变化点数
                    max_n_bkps = max(2, min(10, len(values) // 10))
                    change_points = algo.predict(n_bkps=max_n_bkps)
                elif ruptures_algorithm == 'window':
                    algo = rpt.Window(width=min(40, len(values) // 2), model=ruptures_model).fit(signal)
                    change_points = algo.predict(pen=pen)
                else:
                    # 默认使用 Pelt
                    algo = rpt.Pelt(model=ruptures_model).fit(signal)
                    change_points = algo.predict(pen=pen)
                
                if len(change_points) > 1 and change_points[-1] == len(values):
                    change_points = change_points[:-1]
                
                if len(change_points) == 0:
                    return []
                
                # 分析段
                segments = []
                prev_cp = 0
                for cp in change_points:
                    if cp > prev_cp and cp <= len(values):
                        segment_values = values[prev_cp:cp]
                        if len(segment_values) > 0:
                            segments.append({
                                'start_idx': prev_cp,
                                'end_idx': cp - 1,
                                'mean': np.mean(segment_values),
                                'max': np.max(segment_values),
                                'length': len(segment_values)
                            })
                        prev_cp = cp
                
                if prev_cp < len(values):
                    segment_values = values[prev_cp:]
                    if len(segment_values) > 0:
                        segments.append({
                            'start_idx': prev_cp,
                            'end_idx': len(values) - 1,
                            'mean': np.mean(segment_values),
                            'max': np.max(segment_values),
                            'length': len(segment_values)
                        })
                
                # 识别剧烈变化 - 基于相对变化而非固定阈值
                if len(segments) >= 2:
                    # 计算整个时间序列的基线统计信息（用于动态阈值）
                    overall_mean = np.mean(values)
                    overall_std = np.std(values)
                    baseline_value = overall_mean
                    
                    # 如果标准差很小，使用均值作为基线；否则使用第一个segment的均值作为基线
                    if overall_std < overall_mean * 0.1:  # 标准差小于均值的10%，认为数据相对稳定
                        baseline_value = overall_mean
                    else:
                        # 使用前几个segments的均值作为基线（排除可能的异常段）
                        baseline_segments = segments[:min(3, len(segments))]
                        baseline_values = [seg['mean'] for seg in baseline_segments]
                        baseline_value = np.mean(baseline_values)
                    
                    for i in range(1, len(segments)):
                        prev_seg = segments[i-1]
                        curr_seg = segments[i]
                        
                        # 获取基线参数（根据指标类型动态设置）
                        min_baseline_threshold, reference_value = get_baseline_params(kpi_name)
                        
                        # 计算相对变化百分比（改进版：处理小基线值的情况）
                        if prev_seg['mean'] >= min_baseline_threshold:
                            relative_change_pct = abs((curr_seg['mean'] - prev_seg['mean']) / prev_seg['mean'] * 100)
                        else:
                            # 前一个segment的均值很小，使用绝对变化
                            absolute_change = abs(curr_seg['mean'] - prev_seg['mean'])
                            relative_change_pct = (absolute_change / reference_value) * 100 if reference_value > 0 else 0
                        
                        # 计算相对于基线的偏离百分比（改进版：处理小基线值的情况）
                        if baseline_value >= min_baseline_threshold:
                            baseline_deviation_pct = abs((curr_seg['mean'] - baseline_value) / baseline_value * 100)
                        else:
                            # 基线值很小，使用绝对变化
                            absolute_change = abs(curr_seg['mean'] - baseline_value)
                            baseline_deviation_pct = (absolute_change / reference_value) * 100 if reference_value > 0 else 0
                        
                        # 动态阈值：基于时间窗口内的统计特性
                        # 1. 相对变化阈值：segment之间的相对变化
                        # 2. 基线偏离阈值：相对于基线的偏离
                        # 3. 标准差倍数：如果变化超过多个标准差，认为是异常
                        
                        # 计算变化的标准差倍数
                        if overall_std > 0:
                            change_in_std = abs(curr_seg['mean'] - prev_seg['mean']) / overall_std
                        else:
                            change_in_std = 0
                        
                        # 动态判断是否为异常：
                        # 1. 相对变化 > 50%（segment之间变化超过50%）
                        # 2. 或者相对变化 > 30% 且 基线偏离 > 50%（变化明显且偏离基线）
                        # 3. 或者变化超过2个标准差（统计显著）
                        # 4. 或者相对变化 > 100%（变化超过一倍）
                        
                        is_anomaly = False
                        if relative_change_pct > 100:
                            # 变化超过一倍，肯定是异常
                            is_anomaly = True
                        elif relative_change_pct > 50:
                            # 变化超过50%，认为是异常
                            is_anomaly = True
                        elif relative_change_pct > 30 and baseline_deviation_pct > 50:
                            # 变化超过30%且偏离基线超过50%
                            is_anomaly = True
                        elif change_in_std > 2.0:
                            # 变化超过2个标准差，统计显著
                            is_anomaly = True
                        
                            if is_anomaly:
                                # 使用相对变化百分比和基线偏离百分比中的较大值
                                deviation_pct = max(relative_change_pct, baseline_deviation_pct)
                                
                                change_idx = curr_seg['start_idx']
                                if change_idx < len(datetime_series):
                                    # 直接从DataFrame获取时间戳，保留时区信息
                                    change_time = datetime_series.iloc[change_idx]
                            anomalies.append({
                                'component_name': component,
                                'faulty_kpi': kpi_name,
                                'fault_start_time': to_iso_shanghai(change_time),
                                'severity_score': calculate_severity(deviation_pct, curr_seg['max']),
                                'deviation_pct': float(deviation_pct),  # 确保是Python float类型
                                'method': 'ruptures',
                                'change_idx': int(change_idx)  # 确保是Python int类型
                            })
            except Exception:
                pass
            
            return anomalies
        
        # 使用Z-score检测 - 改进版：基于滑动窗口基线
        def detect_with_zscore(component: str, kpi_name: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
            if len(data) < min_data_points_zscore:
                return []
            
            data = data.sort_values('datetime').reset_index(drop=True)
            values = data['value'].values.astype(float)
            # 直接从DataFrame获取datetime列，保留时区信息
            datetime_series = data['datetime']
            
            anomalies = []
            try:
                # 方法1：全局Z-score检测（原有方法）
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if std_val == 0:
                    return []
                
                z_scores = np.abs((values - mean_val) / std_val)
                anomaly_indices_global = np.where(z_scores > z_threshold)[0]
                
                # 方法2：滑动窗口基线检测（改进版：避免基线窗口污染）
                # 使用前10-15%的数据作为基线窗口（更小的窗口，减少异常值污染）
                baseline_window_size = max(5, min(int(len(values) * 0.15), int(len(values) * 0.1)))
                baseline_values_raw = values[:baseline_window_size]
                
                # 使用IQR方法排除基线窗口中的异常值
                if len(baseline_values_raw) >= 4:
                    q1 = np.percentile(baseline_values_raw, 25)
                    q3 = np.percentile(baseline_values_raw, 75)
                    iqr = q3 - q1
                    if iqr > 0:
                        # 排除超出1.5*IQR范围的异常值
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        baseline_values = baseline_values_raw[
                            (baseline_values_raw >= lower_bound) & (baseline_values_raw <= upper_bound)
                        ]
                    else:
                        baseline_values = baseline_values_raw
                else:
                    baseline_values = baseline_values_raw
                
                # 如果排除异常值后基线窗口太小，使用原始值
                if len(baseline_values) < 3:
                    baseline_values = baseline_values_raw
                
                baseline_mean = np.mean(baseline_values)
                baseline_std = np.std(baseline_values)
                
                # 如果基线标准差为0或太小，使用MAD（Median Absolute Deviation）作为替代
                if baseline_std == 0 or baseline_std < baseline_mean * 0.01:
                    # 使用MAD：median(|x - median(x)|)
                    baseline_median = np.median(baseline_values)
                    mad = np.median(np.abs(baseline_values - baseline_median))
                    # MAD的标准化：对于正态分布，MAD ≈ 0.6745 * std，所以 std ≈ MAD / 0.6745
                    baseline_std = mad / 0.6745 if mad > 0 else std_val
                
                if baseline_std == 0:
                    baseline_std = std_val  # 如果基线标准差仍为0，使用全局标准差
                
                # 计算相对于基线的Z-score
                baseline_z_scores = np.abs((values - baseline_mean) / baseline_std) if baseline_std > 0 else np.zeros_like(values)
                anomaly_indices_baseline = np.where(baseline_z_scores > z_threshold)[0]
                
                # 合并两种方法的异常索引
                anomaly_indices = np.unique(np.concatenate([anomaly_indices_global, anomaly_indices_baseline]))
                
                if len(anomaly_indices) == 0:
                    return []
                
                # 连续性检查
                continuous_segments = []
                current_segment = []
                
                for idx in sorted(anomaly_indices):
                    if not current_segment:
                        current_segment.append(idx)
                    elif idx == current_segment[-1] + 1:
                        current_segment.append(idx)
                    else:
                        if len(current_segment) >= min_consecutive:
                            continuous_segments.append(current_segment.copy())
                        current_segment = [idx]
                
                if len(current_segment) >= min_consecutive:
                    continuous_segments.append(current_segment)
                
                # 分析每个连续异常段
                for segment in continuous_segments:
                    if len(segment) > 0:
                        segment_values = values[segment]
                        
                        segment_mean = np.mean(segment_values)
                        
                        # 获取基线参数（根据指标类型动态设置）
                        min_baseline_threshold, reference_value = get_baseline_params(kpi_name)
                        
                        # 计算相对于基线的偏离百分比（改进版：处理小基线值的情况）
                        # 对于绝对值很小的指标（如CPULoad），当基线值很小时，使用绝对变化而不是相对变化
                        # 设置最小基线值阈值，避免除以接近0的值导致误报
                        # 注意：这里使用的baseline_mean是改进后的基线均值（已排除异常值）
                        if baseline_mean >= min_baseline_threshold:
                            # 基线值足够大，使用相对百分比
                            deviation_pct = abs((segment_mean - baseline_mean) / baseline_mean * 100)
                        elif mean_val >= min_baseline_threshold:
                            # 如果基线均值太小，使用全局均值（如果全局均值也足够大）
                            deviation_pct = abs((segment_mean - mean_val) / mean_val * 100)
                        else:
                            # 基线值和全局均值都很小，使用绝对变化
                            # 将绝对变化转换为等效的百分比（基于指标类型的参考值）
                            absolute_change = abs(segment_mean - baseline_mean)
                            deviation_pct = (absolute_change / reference_value) * 100 if reference_value > 0 else 0
                        
                        # 动态阈值：基于相对变化而非固定阈值
                        # 1. 偏离基线超过50%（相对变化明显）
                        # 2. 或者偏离基线超过30%且Z-score很高（统计显著）
                        # 3. 或者偏离基线超过100%（变化超过一倍）
                        
                        segment_z_score = np.mean(baseline_z_scores[segment]) if len(segment) > 0 else 0
                        
                        is_anomaly = False
                        if deviation_pct > 100:
                            # 变化超过一倍，肯定是异常
                            is_anomaly = True
                        elif deviation_pct > 50:
                            # 偏离基线超过50%，认为是异常
                            is_anomaly = True
                        elif deviation_pct > 30 and segment_z_score > z_threshold:
                            # 偏离基线超过30%且Z-score超过阈值
                            is_anomaly = True
                        
                        if is_anomaly:
                            max_value = np.max(segment_values)
                            # 直接从DataFrame获取时间戳，保留时区信息
                            segment_start_idx = segment[0]
                            if segment_start_idx < len(datetime_series):
                                segment_time = datetime_series.iloc[segment_start_idx]
                            else:
                                segment_time = datetime_series.iloc[0]  # fallback
                            
                            anomalies.append({
                                'component_name': component,
                                'faulty_kpi': kpi_name,
                                'fault_start_time': to_iso_shanghai(segment_time),
                                'severity_score': calculate_severity(deviation_pct, max_value),
                                'deviation_pct': float(deviation_pct),  # 确保是Python float类型
                                'method': 'zscore',
                                'change_idx': int(segment[0])  # 确保是Python int类型
                            })
            except Exception:
                pass
            
            return anomalies
        
        # 基于绝对阈值的检测（用于检测持续高值）
        def detect_with_absolute_threshold(component: str, kpi_name: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
            """
            检测持续高值，即使没有变化点
            
            改进逻辑：
            1. 使用基线窗口（前30%数据）来判断是否为正常状态
            2. 如果基线窗口也是高值且整个时间窗口稳定，可能是正常状态，不报告异常
            3. 只有当从低值变化到高值，或者基线窗口低但后续有高值段时，才报告异常
            """
            abs_threshold = get_absolute_threshold(kpi_name)
            if abs_threshold is None:
                return []
            
            if len(data) < min_data_points_zscore:
                return []
            
            data = data.sort_values('datetime').reset_index(drop=True)
            values = data['value'].values.astype(float)
            # 直接从DataFrame获取datetime列，保留时区信息
            datetime_series = data['datetime']
            
            # 使用前10-15%的数据作为基线窗口（更小的窗口，减少异常值污染）
            baseline_window_size = max(5, min(int(len(values) * 0.15), int(len(values) * 0.1)))
            baseline_values_raw = values[:baseline_window_size]
            
            # 使用IQR方法排除基线窗口中的异常值
            if len(baseline_values_raw) >= 4:
                q1 = np.percentile(baseline_values_raw, 25)
                q3 = np.percentile(baseline_values_raw, 75)
                iqr = q3 - q1
                if iqr > 0:
                    # 排除超出1.5*IQR范围的异常值
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    baseline_values = baseline_values_raw[
                        (baseline_values_raw >= lower_bound) & (baseline_values_raw <= upper_bound)
                    ]
                else:
                    baseline_values = baseline_values_raw
            else:
                baseline_values = baseline_values_raw
            
            # 如果排除异常值后基线窗口太小，使用原始值
            if len(baseline_values) < 3:
                baseline_values = baseline_values_raw
            
            baseline_mean = np.mean(baseline_values)
            baseline_std = np.std(baseline_values)
            
            # 计算整个时间窗口的统计信息
            overall_mean = np.mean(values)
            overall_std = np.std(values)
            
            # 改进1：如果基线窗口也是高值，且整个时间窗口稳定（标准差小），可能是正常状态
            # 判断是否为稳定状态：标准差小于均值的5%或小于阈值的5%
            stability_threshold = min(overall_mean * 0.05, abs_threshold * 0.05)
            is_stable = overall_std < stability_threshold
            
            # 如果基线窗口平均值也超过阈值，且整个时间窗口稳定，可能是正常状态
            if baseline_mean > abs_threshold and is_stable:
                # 检查是否所有数据点都超过阈值
                if np.all(values > abs_threshold):
                    # 整个时间窗口都是高值且稳定，可能是组件的正常状态，不报告异常
                    return []
            
            # 改进2：对于CPU等快速变化的指标，允许接近阈值的数据点也算作异常段的一部分
            # 对于CPU指标，使用"软阈值"：阈值*0.875（即80%阈值的87.5% = 70%）
            # 这样可以捕获71.37%这样的接近阈值的高值
            is_cpu_metric = 'CPU' in kpi_name or 'CPULoad' in kpi_name
            # 对于CPU指标，使用更低的软阈值（70%），以便捕获接近阈值的高值
            soft_threshold = abs_threshold * 0.875 if is_cpu_metric else abs_threshold
            
            # 检查是否有值超过绝对阈值或软阈值
            high_value_indices = np.where(values > abs_threshold)[0]
            near_threshold_indices = np.where((values > soft_threshold) & (values <= abs_threshold))[0] if is_cpu_metric else np.array([])
            
            # 合并超过阈值和接近阈值的数据点
            if len(near_threshold_indices) > 0:
                all_high_indices = np.unique(np.concatenate([high_value_indices, near_threshold_indices]))
            else:
                all_high_indices = high_value_indices
            
            if len(all_high_indices) == 0:
                return []
            
            # 改进3：对于CPU指标，降低min_consecutive要求（从3降到1）
            # 因为CPU可能快速变化，只有1-2个点超过阈值
            # 如果有一个点超过绝对阈值，即使前后没有接近阈值的高值点，也应该认为是异常
            effective_min_consecutive = 1 if is_cpu_metric else min_consecutive
            
            # 改进4：只有当从低值变化到高值时，才报告异常
            # 如果基线窗口低于阈值，但后续有高值段，则报告异常
            # 或者，如果基线窗口高于阈值，但后续有更高的值段（显著上升），也报告异常
            
            # 找到连续的高值段（包括接近阈值的数据点）
            continuous_segments = []
            current_segment = []
            
            for idx in sorted(all_high_indices):
                if not current_segment:
                    current_segment.append(idx)
                elif idx == current_segment[-1] + 1:
                    current_segment.append(idx)
                else:
                    if len(current_segment) >= effective_min_consecutive:
                        continuous_segments.append(current_segment.copy())
                    current_segment = [idx]
            
            if len(current_segment) >= effective_min_consecutive:
                continuous_segments.append(current_segment)
            
            anomalies = []
            for segment in continuous_segments:
                if len(segment) > 0:
                    segment_values = values[segment]
                    segment_mean = np.mean(segment_values)
                    max_value = np.max(segment_values)
                    
                    # 改进3：检查是否是从低值变化到高值
                    # 如果segment开始前有数据点，检查前一个点的值
                    segment_start_idx = segment[0]
                    if segment_start_idx > 0:
                        prev_value = values[segment_start_idx - 1]
                        # 如果前一个值低于阈值，说明是从低值变化到高值，这是异常
                        is_change_from_low = prev_value <= abs_threshold
                    else:
                        # segment从开始就有，检查基线窗口
                        is_change_from_low = baseline_mean <= abs_threshold
                    
                    # 改进4：如果基线窗口也是高值，检查是否有显著上升
                    # 如果segment均值显著高于基线均值（超过10%），也认为是异常
                    if baseline_mean > abs_threshold:
                        relative_increase = (segment_mean - baseline_mean) / baseline_mean * 100
                        is_significant_increase = relative_increase > 10.0
                    else:
                        is_significant_increase = False
                    
                    # 只有当从低值变化到高值，或者有显著上升时，才报告异常
                    if not (is_change_from_low or is_significant_increase):
                        # 如果基线窗口也是高值，且没有显著变化，可能是正常状态，跳过
                        continue
                    
                    # 计算正常基线值（用于计算偏离百分比）
                    # 优先使用segment开始前的正常值，如果没有则使用基线窗口中低于阈值的数据点
                    normal_baseline = None
                    
                    # 方法1：使用segment开始前的正常值（如果存在）
                    if segment_start_idx > 0:
                        # 检查segment开始前的数据点，找到最后一个低于阈值的数据点
                        prev_normal_values = []
                        for i in range(segment_start_idx - 1, -1, -1):
                            if values[i] <= abs_threshold:
                                prev_normal_values.append(values[i])
                            if len(prev_normal_values) >= 3:  # 至少3个正常值
                                break
                        
                        if len(prev_normal_values) > 0:
                            normal_baseline = np.mean(prev_normal_values)
                    
                    # 方法2：如果方法1没有找到正常值，使用基线窗口中低于阈值的数据点
                    if normal_baseline is None:
                        baseline_normal_values = baseline_values[baseline_values <= abs_threshold]
                        if len(baseline_normal_values) > 0:
                            normal_baseline = np.mean(baseline_normal_values)
                    
                    # 方法3：如果方法2也没有找到，使用整个时间窗口中低于阈值的数据点（排除异常段）
                    if normal_baseline is None:
                        # 排除所有异常段的数据点
                        all_normal_indices = []
                        for i in range(len(values)):
                            if values[i] <= abs_threshold:
                                # 检查是否在异常段中
                                is_in_segment = False
                                for seg in continuous_segments:
                                    if i in seg:
                                        is_in_segment = True
                                        break
                                if not is_in_segment:
                                    all_normal_indices.append(i)
                        
                        if len(all_normal_indices) > 0:
                            normal_baseline = np.mean(values[all_normal_indices])
                    
                    # 方法4：如果以上方法都没有找到正常值，使用阈值作为参考
                    if normal_baseline is None or normal_baseline <= 0:
                        normal_baseline = abs_threshold
                    
                    # 计算相对于正常基线的偏离百分比（更能反映实际异常程度）
                    # 如果正常基线值很小，使用get_baseline_params的逻辑来处理
                    min_baseline_threshold, reference_value = get_baseline_params(kpi_name)
                    
                    # 判断是否使用相对百分比还是绝对变化转换
                    # 1. 如果正常基线值小于最小阈值，使用绝对变化转换
                    # 2. 如果正常基线值相对于参考值很小（<10%），也使用绝对变化转换
                    #    这样可以避免当正常基线值很小时，相对百分比异常大的问题
                    # 3. 否则使用相对百分比（更能反映实际异常程度）
                    use_relative = (normal_baseline >= min_baseline_threshold and 
                                   normal_baseline >= reference_value * 0.1)
                    
                    if use_relative:
                        # 正常基线值足够大，使用相对百分比
                        deviation_pct = abs((segment_mean - normal_baseline) / normal_baseline * 100)
                    else:
                        # 正常基线值很小，使用绝对变化转换为百分比
                        absolute_change = abs(segment_mean - normal_baseline)
                        deviation_pct = (absolute_change / reference_value) * 100 if reference_value > 0 else 0
                    
                    # 如果平均值超过阈值，认为是异常
                    if segment_mean > abs_threshold:
                        # 直接从DataFrame获取时间戳，保留时区信息
                        if segment_start_idx < len(datetime_series):
                            segment_time = datetime_series.iloc[segment_start_idx]
                        else:
                            segment_time = datetime_series.iloc[0]  # fallback
                        
                        anomalies.append({
                            'component_name': component,
                            'faulty_kpi': kpi_name,
                            'fault_start_time': to_iso_shanghai(segment_time),
                            'severity_score': calculate_severity(deviation_pct, max_value),
                            'deviation_pct': float(deviation_pct),  # 确保是Python float类型
                            'method': 'absolute_threshold',
                            'change_idx': int(segment_start_idx)  # 确保是Python int类型
                        })
            
            return anomalies
        
        # 检测异常
        all_anomalies = []
        grouped = df_core.groupby(['cmdb_id', 'kpi_name'])
        
        for (component, kpi_name), group in grouped:
            if len(group) < min(min_data_points_ruptures, min_data_points_zscore):
                continue
                
            anomalies = []
            
            # 首先尝试基于绝对阈值的检测（用于检测持续高值）
            absolute_anomalies = detect_with_absolute_threshold(component, kpi_name, group)
            anomalies.extend(absolute_anomalies)
            
            if method in ['ruptures', 'both']:
                ruptures_anomalies = detect_with_ruptures(component, kpi_name, group)
                anomalies.extend(ruptures_anomalies)
            
            if method in ['zscore', 'both']:
                zscore_anomalies = detect_with_zscore(component, kpi_name, group)
                anomalies.extend(zscore_anomalies)
            
            # 如果使用both方法，去重（保留第一个检测到的）
            if method == 'both' and len(anomalies) > 1:
                anomalies = sorted(anomalies, key=lambda x: x.get('change_idx', 0))
                ruptures_results = [a for a in anomalies if a['method'] == 'ruptures']
                if ruptures_results:
                    anomalies = [ruptures_results[0]]
                else:
                    anomalies = [anomalies[0]]
            
            all_anomalies.extend(anomalies)
        
        # 去重：同一个组件-指标组合只保留一个异常（选择最早的）
        seen = {}
        for anomaly in all_anomalies:
            key = (anomaly['component_name'], anomaly['faulty_kpi'])
            if key not in seen:
                seen[key] = anomaly
            else:
                existing_time = pd.to_datetime(seen[key]['fault_start_time'])
                current_time = pd.to_datetime(anomaly['fault_start_time'])
                if current_time < existing_time:
                    seen[key] = anomaly
        
        final_anomalies = list(seen.values())
        
        # 按照偏离程度从高到低排序
        final_anomalies = sorted(final_anomalies, key=lambda x: x.get('deviation_pct', 0), reverse=True)
        
        # 如果指定了top参数且method不是both，限制返回数量
        if method != 'both' and top > 0:
            final_anomalies = final_anomalies[:top]
        
        # 转换numpy/pandas类型为Python原生类型，以便JSON序列化
        def convert_to_native_types(obj):
            """递归转换numpy/pandas类型为Python原生类型"""
            # 检查numpy整数类型
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            # 检查numpy浮点类型
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            # 检查numpy数组
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # 检查pandas的NA值
            elif pd.isna(obj):
                return None
            # 检查字典
            elif isinstance(obj, dict):
                return {key: convert_to_native_types(value) for key, value in obj.items()}
            # 检查列表和元组
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native_types(item) for item in obj]
            # 其他类型直接返回
            else:
                return obj
        
        # 转换所有异常数据
        final_anomalies = [convert_to_native_types(anomaly) for anomaly in final_anomalies]
        
        return json.dumps(final_anomalies, ensure_ascii=False, indent=2)
