# RCA Agent（根因分析智能体）

RCA Agent 是一个面向分布式系统的自动化根因分析系统，结合指标（Metrics）、调用链（Traces）、日志（Logs）三类数据源，采用多智能体协同和可迭代评估决策流程，定位故障组件、故障发生时间与根因解释。项目同时提供因果图（Causal Graph）与结构化因果模型（SCM）管线，以及 CPU 火焰图分析能力。

## 核心能力
- 多智能体协同：指标故障分析、根因定位、评估决策三个子代理由协调器串联、支持迭代优化
- 数据工具集成：本地 OpenRCA 数据集的指标、日志、调用链分析工具
- 因果分析管线：数据预处理 → 因果图构建与优化 → SCM 训练 → 根因推断
- 交互式流式分析：命令行交互、实时输出、可选接入 Langfuse 追踪
- CPU 火焰图：可选采集与分析，辅助定位性能瓶颈

## 数据集准备
- 默认使用 OpenRCA Bank 场景数据，路径为 datasets/OpenRCA/Bank
- 将数据集放到上述路径即可，无需额外代码配置
- 示例目录结构：

```text
datasets/
└── OpenRCA/
    └── Bank/
        └── telemetry/
            └── 2021_03_04/
                ├── metric/
                │   ├── metric_app.csv
                │   └── metric_container.csv
                ├── log/
                │   └── log_service.csv
                └── trace/
                    └── trace_span.csv
```

## 安装与环境
- Python 版本：>= 3.11, < 3.14（见 [pyproject.toml](file:///Users/xuhonggong/Desktop/files/project/huawei/RCAAgent/pyproject.toml)）
- 依赖管理：Poetry（推荐）
  - 安装依赖：在项目根目录执行

```bash
poetry install
```

- 运行示例脚本时使用 Poetry

```bash
poetry run python examples/example_rca_scenario.py
```

## 快速开始（交互式 RCA）
- 运行交互式 RCA 场景分析（流式输出）

```bash
poetry run python examples/example_rca_scenario.py
```

- 按提示输入时间范围与故障数量；系统将：
  - 指标故障分析：识别异常组件与故障开始时间
  - 根因定位：结合日志/调用链补证并产出根因解释
  - 评估决策：并行多评估代理综合判定，必要时迭代优化
- 结果会保存到 outputs/rca

## 磁盘故障注入与日志定位（Disk Fault）

本项目内置了一个磁盘故障注入脚本，用于在 Linux 环境下通过 `scsi_debug` / `dmsetup` / `fio` 等工具模拟磁盘故障，并采集 `kernel/syslog/app` 三类日志，生成可被本项目直接加载分析的数据集。

### 1) 运行故障注入脚本

脚本位置：[disk_fault_injector.py](file:///Users/xuhonggong/Desktop/files/project/huawei/RCAAgent/examples/disk_fault_injector.py)

- 依赖（需要具备 root 权限）：
  - `modprobe`, `lsblk`, `mount`, `umount`, `mkfs.ext4`, `fio`, `dd`, `truncate`, `blockdev`
  - 场景 `bad_disk` 额外需要 `dmsetup`；采集内核日志建议安装 `journalctl`

- 典型用法：直接把输出目录写到 `datasets/disk_fault_logs`，后续可直接在 RCA 中选择 Disk Fault 分析

```bash
sudo python3 examples/disk_fault_injector.py \
  --output-dir datasets/disk_fault_logs \
  --interval-seconds 600 \
  --cycles 3 \
  --output-layout daily \
  --noise-mode logger
```

- 参数说明（常用）：
  - `--scenario {bad_disk,slow_disk,pressure}`：指定注入场景；不指定则每个 cycle 随机选择
  - `--interval-seconds`：每次注入窗口时长（秒）
  - `--cycles`：执行次数（0 表示一直跑直到 Ctrl+C）
  - `--output-dir`：输出目录（建议设置为 `datasets/disk_fault_logs`）
  - `--output-layout daily`：按天聚合输出（默认），目录名为 `YYYY-MM-DD`
  - `--dry-run`：仅打印将执行的命令，不做真实注入

### 2) 输出数据结构（与 DataLoader 对齐）

默认 `--output-layout daily` 时，脚本会按天落盘并追加写入：

```text
datasets/disk_fault_logs/
└── 2026-01-12/
    ├── app.log
    ├── kernel.log
    ├── syslog.log
    └── run.log
datasets/disk_fault_logs/fault_injection_record.csv
```

- `app.log/kernel.log/syslog.log`：每行以 UTC ISO 时间戳开头（脚本会做归一化），后面是原始消息
- `run.log`：每个窗口的执行信息（包含 window / fault_type / trigger_delay 等）
- `fault_injection_record.csv`：每次注入窗口的 `start_utc/end_utc/trigger_utc/fault_type`，可用作“地面真值”快速缩小定位时间窗

### 3) 用生成的日志做故障定位（推荐流程）

- Step A：用 `fault_injection_record.csv` 快速确定分析时间窗
  - 以 `trigger_utc` 为中心，建议先取前后 15 分钟作为初始分析窗口

- Step B：用交互式 RCA 进行定位（Disk Fault 域）
  - 运行交互式分析脚本并选择 `Disk Fault (System Logs)` 场景

```bash
poetry run python examples/example_rca_scenario.py
```

  - 输入时间范围（Disk Fault 场景的时间与日志时间均为 UTC；可直接使用 `fault_injection_record.csv` 中的时间）
  - 系统会主要通过日志工具从 `kernel/syslog/app` 三类日志中提取证据并输出根因组件、故障开始时间与原因解释

  - 建议优先关注的日志证据：
    - `kernel.log`：I/O error、timeout、blocked tasks、EXT4 错误、scsi_debug/dm-* 相关报错
    - `syslog.log`：mount/umount/mkfs 失败、服务层报错、系统告警
    - `app.log`：业务侧延迟/失败（脚本可生成噪声与压力写入日志，便于验证定位链路）

## 非交互脚本
- 因果管线（推荐用于离线训练 + 推断）：
  - 预处理数据

```bash
poetry run python examples/prepare_causal_data.py
```

  - 构建并优化因果图（可选算法：pc/pcmci/granger/varlingam/granger_pc）

```bash
poetry run python examples/build_causal_graph.py
```

  - 训练 SCM 模型

```bash
poetry run python examples/train_causal_model.py
```

  - 基于 SCM 进行根因分析

```bash
poetry run python examples/run_root_cause_analysis.py
```

## 标准数据格式（RCA 通用 Schema）
- 目标：在兼容 OpenRCA 的基础上，提供通用、可扩展的数据结构，统一三类数据（metric/trace/log）的字段与类型，并在数据加载层强制校验与规范化；遇到不合法数据将抛出异常。
- 适用：所有工具均基于统一 Schema；数据加载基类在返回数据前执行必填项与类型校验。不进行任何数据集兼容性转换；数据集特有字段映射由具体 DataLoader 子类负责（OpenRCADataLoader 完成 OpenRCA → 通用字段的转换）。

### Metric（统一字段）
| 字段名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| timestamp | Int64 | 是 | 毫秒时间戳（统一过滤依据） |
| entity_id | str | 是 | 资源/组件标识 |
| metric_name | str | 是 | 指标名称 |
| value | float | 是 | 指标数值 |
说明：时间展示通过工具层按 DataLoader 时区格式化。

### Log（统一字段）
| 字段名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| timestamp | Int64 | 是 | 毫秒时间戳（统一过滤依据） |
| entity_id | str | 是 | 服务/组件名称 |
| message | str | 是 | 原始日志文本 |
| severity | Optional[str] | 否 | 日志等级（可选） |
说明：只保留单一时间戳列；需要可读时间时在工具层按 DataLoader 时区格式化。

### Trace（统一字段）
| 字段名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| timestamp | Int64 | 是 | 毫秒时间戳（span开始时间） |
| entity_id | str | 是 | 服务/组件名称 |
| trace_id | str | 是 | Trace 唯一标识 |
| span_id | str | 是 | Span 唯一标识 |
| parent_span_id | Optional[str] | 否 | 父 Span 标识 |
| duration_ms | float | 是 | Span 持续时间（毫秒，非负） |
| status_code | Optional[str] | 否 | 状态码（预留字段） |
说明：统一仅保留 timestamp，避免多时间列混用；时间展示按 DataLoader 时区格式化。

### 使用方式
- 统一 DataFrame 接口（推荐在工具层直接使用）：
  - Metrics: BaseDataLoader.get_metrics(start_time, end_time)
  - Logs: BaseDataLoader.get_logs(start_time, end_time)
  - Traces: BaseDataLoader.get_traces(start_time, end_time)
- 策略：BaseDataLoader 只做字段校验与类型规范；数据集字段映射由具体 DataLoader 子类负责（OpenRCADataLoader 完成 OpenRCA → 通用字段的转换）。

## MetricSemanticAdapter 接口说明

为了将数据集与领域语义从通用异常检测逻辑中解耦，Metric 工具通过适配器接口提供“语义注入点”。适配器负责派生指标构建、指标分类、绝对阈值与基线参数提供、严重级文本格式化等。通用检测流程仅调用这些接口。

### 接口列表与使用场景

- get_candidate_entities(df, time_range=None, label_selector=None) -> List[str]
  - 作用：返回候选实体 ID 列表，用于缩小分析范围
  - 入参：df（统一 Schema 指标表）、time_range（可选 {"start_time","end_time"}）、label_selector（可选字典）
  - 出参：候选实体 ID 列表
  - 场景：数据量较大时预筛选；结合场景优先级或标签过滤
  - 参考实现：OpenRCAMetricAdapter.get_candidate_entities

- build_derived_metrics(df) -> Optional[pd.DataFrame]
  - 作用：构造派生指标（例如根据 HeapMemoryMax 与 HeapMemoryUsed 构造 JVM_Heap_Usage）
  - 入参：df（统一 Schema）
  - 出参：派生指标 DataFrame（同 Schema），若无派生则返回 None
  - 场景：原始数据未直接提供关键指标，需要通过组合计算得到
  - 参考实现：OpenRCAMetricAdapter.build_derived_metrics

- classify_metric(metric_name) -> Optional[Dict[str, Any]]
  - 作用：将指标名称映射为语义类别与单位，用于后续阈值与基线策略
  - 入参：metric_name（字符串）
  - 出参：字典，如 {"kind": "cpu.util", "unit": "percent"}；不识别则返回 None
  - 场景：筛选“核心”指标；按类别选择检测策略
  - 参考实现：OpenRCAMetricAdapter.classify_metric

- get_absolute_threshold(metric_class) -> Optional[float]
  - 作用：提供绝对阈值，用于持续高值检测（即使没有明显变点）
  - 入参：metric_class（上一步 classify_metric 的返回）
  - 出参：阈值（float），若该指标不适用绝对阈值则返回 None
  - 场景：CPU/内存/网络利用率等场景下的阈值告警
  - 参考实现：OpenRCAMetricAdapter.get_absolute_threshold

- get_baseline_params(metric_class) -> Optional[Dict[str, float]]
  - 作用：提供基线语义参数，用于在基线很小或近零时选择参考值计算偏差
  - 入参：metric_class（上一步 classify_metric 的返回）
  - 出参：{"min_baseline_threshold": float, "reference_value": float}
  - 场景：小基线指标（如 JVM CPULoad、Heap 使用率）避免除以接近 0 的值导致误报
  - 参考实现：OpenRCAMetricAdapter.get_baseline_params

- format_severity(deviation_pct, max_value) -> Optional[str]
  - 作用：将偏离百分比与最大值转换为可读的严重级描述，便于报告呈现
  - 入参：deviation_pct（偏离百分比）、max_value（该段最大值）
  - 出参：文本描述（如“严重（最大值：xx.x，偏离：yy.y%）”），或 None
  - 场景：对外显示异常结果的严重程度
  - 参考实现：OpenRCAMetricAdapter.format_severity

### 适配器实现示例
- 默认适配器：OpenRCAMetricAdapter（已注册为 openrca）
  - 位置：metric_adapter.py: OpenRCAMetricAdapter
  - 行为：提供 OpenRCA 数据集的候选实体列表、JVM 派生指标、核心指标分类、绝对阈值与基线参数、严重级文本
- 空适配器：NullMetricAdapter
  - 位置：metric_adapter.py: NullMetricAdapter
  - 行为：返回通用缺省值；用于未注册或未指定适配器时的降级

### 如何配置与使用
- 在 Metric 工具初始化时按配置创建适配器：见 metric_tool.py: initialize
- 配置示例：{"metric_adapter": "openrca", "dataset_path": "datasets/OpenRCA/Bank"}
- 自定义适配器流程：
  1. 新增类继承 MetricSemanticAdapter 并实现上述接口
  2. 调用 register_metric_adapter("your_name", lambda: YourAdapter())
  3. 在配置中设置 "metric_adapter": "your_name"
  
## Agent 工具箱

系统为 Agent 提供了三类核心分析工具，用于从不同维度诊断系统故障。

### 1. Metric 工具 (MetricAnalysisTool)
面向统一实体（Entity）的时间序列指标分析，支持领域语义适配与多算法异常检测。

- **get_available_entities**: 可用实体列表
  - **说明**: 列出时间窗内存在指标数据的实体及其指标数量，便于分析范围预估。
- **get_available_metrics**: 可用指标列表
  - **说明**: 按实体与名称模式（可选）过滤，分组展示可用指标；参数为 `entity_id`（已统一为实体标识）。
- **compare_entity_metrics**: 单实体指标对比（目标 vs 基线）
  - **说明**: 通过稳健统计（mean、p99）对比目标与基线窗口的指标变化，支持名称列表或模式选择。
- **get_metric_statistics**: 指标详细统计
  - **说明**: 输出指定实体与指标的详细分布统计（mean、std、min、max、p25/p50/p75/p95/p99、non_zero_ratio）。
- **find_metric_outliers**: 指标异常值扫描（Z-Score）
  - **算法**: Z-Score（跨实体/指标），支持最小点数与阈值配置
  - **说明**: 输出时间（按 DataLoader 时区格式化）、实体、指标、值与统计信息。
- **detect_metric_anomalies**: 综合异常检测（适配器驱动）
  - **算法**: 变点检测（ruptures: PELT/Binseg/Dynp/Window）、Z-Score 连续点检测、绝对阈值检测
  - **语义**: 通过 MetricSemanticAdapter 提供派生指标（如 JVM_Heap_Usage）、指标分类、绝对阈值与基线参数、严重级文本
  - **行为**: 自动候选实体筛选（若未显式传入 `entities`）、仅保留适配器识别的“核心”指标、时区统一由 DataLoader 提供
  - **返回**: JSON 列表，包含 component_name（实体ID）、faulty_kpi、fault_start_time（ISO，按时区格式化）、severity_score、deviation_pct、method、change_idx

时区与格式化：所有时间输出均通过 DataLoader 的 `get_timezone()` 提供时区，并用工具层 `to_iso_with_tz` 转换。

### 2. Log 工具 (LocalLogAnalysisTool)
主要用于日志模式挖掘与错误统计，帮助定位具体的错误信息与异常日志模式。

- **get_log_summary**: 日志概览
  - **算法**: Keyword Counting (Regex)
  - **功能**: 统计指定时间段内的日志总量、错误/警告日志占比、以及最活跃的服务。
- **extract_log_templates_drain3**: 日志模板挖掘
  - **算法**: Drain3 (Online Log Parsing)
  - **功能**: 从原始日志中提取结构化模板，识别罕见或新增的异常日志模式（Template）。
- **query_logs**: 日志检索
  - **算法**: Regex Search
  - **功能**: 根据实体（entity_id）或正则表达式模式检索具体日志条目；时间展示按 DataLoader 时区格式化。

### 3. Trace 工具 (LocalTraceAnalysisTool)
主要用于分布式调用链分析，通过统计学方法与异常检测算法定位链路上的性能瓶颈。

- **train_iforest_model**: 异常检测模型训练 (Isolation Forest)
  - **算法**: Isolation Forest (Unsupervised Learning), Sliding Window
  - **功能**: 基于历史 Trace 数据，针对每一对服务调用关系（Parent -> Child）训练异常检测模型。
- **detect_anomalies_iforest**: 调用链异常检测 (Isolation Forest)
  - **算法**: Isolation Forest
  - **功能**: 使用预训练模型检测当前时间段内的 Trace 异常，识别偏离正常模式的调用路径；时间展示按 DataLoader 时区格式化。
- **detect_anomalies_zscore**: 延迟统计异常检测 (Z-Score)
  - **算法**: Z-Score (Standard Score)
  - **功能**: 基于正态分布假设，检测延迟显著偏离均值（如 > 3 sigma）的 Span.
- **analyze_trace_call_tree**: 单链路分析
  - **算法**: Graph Traversal (BFS/DFS via NetworkX)
  - **功能**: 重建并可视化特定 Trace ID 的完整调用树，展示关键路径耗时。
- **find_slow_spans**: 慢 Span 检索
  - **算法**: Threshold-based Filtering
  - **功能**: 找出系统中耗时最长的 Span，定位具体慢接口；输出时间按 DataLoader 时区格式化。
- **get_dependency_graph**: 依赖拓扑分析
  - **算法**: Graph Aggregation
  - **功能**: 统计服务间的调用频率与依赖关系，构建服务拓扑视图。
- **identify_bottlenecks**: 系统瓶颈识别
  - **算法**: Duration Aggregation & Impact Analysis
  - **功能**: 计算各服务总耗时占比，识别对系统整体延迟影响最大的瓶颈服务。

## 提示词与领域适配器集成
- 指标故障分析与评估提示词由领域适配器动态注入语义：
  - metric_fault_analyst：使用适配器提供的候选实体与核心指标提示片段
  - evaluation_sub_agent / decision：附加适配器提供的数据集限制说明，强调“区分数据缺失与分析遗漏”
- 位置：
  - 提示构建：src/agents/rca_config.py 的 `get_metric_fault_analyst_prompt`、`get_evaluation_sub_agent_prompt`、`get_evaluation_decision_prompt`
  - 领域适配器：src/agents/domain_adapter.py 的 `DomainPromptAdapter` 与 `get_prompt_hints`

## 迭代优化记录

用于记录每次提示词优化或工具优化对 RCA 定位效果的提升情况。
**注：以下数据并不是在全量数据集上进行测试，而是在预先选出的几个用例上进行稳定性测试，所以在OpenRCA数据集上的平均成功率看着会比较高。**

| 日期 | 优化类型 | 优化内容 | 平均时延 (min) | 平均输入 token | 平均输出 token | 平均成功率 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-12-10 | 第一版本 | 使用python进行log/trace/metric分析 | 18 | 42.25万 | 28404 | 50 |
| 2025-12-10 | metric工具优化 | 引入rupture库以及z-score算法进行metric异常检测 | 14 | 31万 | 22007 | 60 |
| 2025-12-12 | 评估决策 | 增加多维度并行评估和最终决策机制以及多轮迭代 | 43 | 85万 | 64849 | 65 |
| 2026-1-6 | 工具增强，Agent层级优化，删除python执行 | log增加brain3模板提取，trace增加孤立森林算法，metric增加对称比率过滤，Agent层级从三层优化为两层 | 17 | 104万 | 30099 | 70 |
