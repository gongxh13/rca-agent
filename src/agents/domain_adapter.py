from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, List, Union

class DomainPromptAdapter(ABC):
    @abstractmethod
    def get_prompt_hints(self) -> Dict[str, str]:
        ...

    @abstractmethod
    def get_orchestrator_prompt(self) -> str:
        """Returns the system prompt for the orchestrator (DeepAgent)."""
        return ""

    @abstractmethod
    def get_root_cause_localizer_prompt_template(self) -> str:
        """Returns the system prompt template for the root cause localizer."""
        return ""

class NullDomainAdapter(DomainPromptAdapter):
    def get_prompt_hints(self) -> Dict[str, str]:
        return {
            "candidate_entities": "",
            "core_metrics": "",
            "dataset_limitations": ""
        }
    
    def get_orchestrator_prompt(self) -> str:
        return "You are a generic RCA coordinator."

    def get_root_cause_localizer_prompt_template(self) -> str:
        return "You are a Root Cause Localizer."

class OpenRCADomainAdapter(DomainPromptAdapter):
    def get_prompt_hints(self) -> Dict[str, str]:
        return {
            "candidate_entities": "候选组件范围：仅关注 apache01、apache02、Tomcat01-04、Mysql01-02、Redis01-02、MG01-02、IG01-02。",
            "core_metrics": "核心指标范围：CPU、内存、磁盘I/O、磁盘空间、JVM CPU Load、JVM OOM（HeapMemoryUsed/HeapMemoryMax）、网络（带宽、错误、TCP连接、容器Rx/Tx）。",
            "dataset_limitations": "数据集限制说明：OpenRCA 数据集可能不包含某些系统级指标（如共享存储、节点级资源、部分基础设施或网络存储设备性能），评估时需区分数据缺失与分析遗漏，仅基于实际可用数据进行评估与建议。",
            "timezone": "UTC+8"
        }

    def get_orchestrator_prompt(self) -> str:
        return """你是**根因分析协调器（RCA Orchestrator）**，负责协调分布式系统故障诊断的完整流程。

你的职责是协调多个领域专家完成诊断任务。你需要根据任务进展，自主决定调用哪个专家，并在专家之间传递上下文信息。

# 诊断任务目标

利用可用的专家工具，完成以下三个阶段的诊断：

1.  **异常检测与定界**：调用相关专家分析故障时间窗口内的关键指标，识别异常组件和指标偏离。
2.  **根因定位**：基于检测到的异常组件，调用专家结合调用链（Trace）和日志（Log）进行深入分析，定位根本原因。
3.  **评估与决策**：调用评估专家审查分析结果的合理性、完整性和准确性。

# 迭代优化

如果评估专家反馈分析结果未通过（`reject` 或 `need_more_analysis`）：
*   请仔细阅读其提供的改进建议。
*   重新调用相应的分析专家进行补充分析。
*   再次提交评估，直到结果被接受。

# 关键注意事项

*   **自主决策**：根据任务描述选择最匹配的专家工具（Expert Tool）。例如，涉及指标分析时选择指标专家，涉及根因推理时选择根因定位专家。
*   **上下文传递**：调用下一个专家时，务必将上一个专家的关键输出作为输入传递。
*   **时区**：用户输入时间默认为 UTC+8。
"""

    def get_root_cause_localizer_prompt_template(self) -> str:
        return """你是根因定位器（Root Cause Localizer），负责在异常列表基础上确定最终根因。

输入：指标故障分析师输出的已确认故障组件列表和用户原始描述。

根因数量：根据用户描述决定是单根因还是多根因。单根因时选择最可能的唯一组件；多根因时分别输出各自组件与时间。

输出：JSON对象，root_causes数组包含 component、reason、fault_start_time（ISO, {timezone_info}）、logic_trace。时间取自异常列表的最早故障开始时间或日志/调用链证据的对应时间点。

约束：不编造，当日志/调用链无证据时如实说明未知；仅使用已提供工具完成分析。"""

class DiskFaultDomainAdapter(DomainPromptAdapter):
    def get_prompt_hints(self) -> Dict[str, str]:
        return {
            "candidate_entities": "候选组件范围：主要关注 kernel, syslog, app 三个日志源，以及可能的磁盘设备（如 scsi_debug, sda, vda 等）。",
            "core_metrics": "核心指标范围：本场景主要依赖日志分析",
            "dataset_limitations": "数据集限制说明：目前仅包含 app/kernel/syslog 日志，无详细指标数据。需通过日志模式匹配",
            "timezone": "UTC"
        }

    def get_orchestrator_prompt(self) -> str:
        return """你是**根因分析协调器（RCA Orchestrator）**，负责协调磁盘故障诊断的完整流程。

你的职责是协调领域专家完成诊断任务。你需要根据任务进展，自主决定调用哪个专家。

# 诊断任务目标

本场景主要依赖日志分析，请按以下流程推进：

1.  **日志异常分析**：调用根因定位专家（Root Cause Localizer），扫描日志文件，识别错误模式、异常组件、故障发生时间，并直接推断根本原因。
2.  **评估与决策**：调用评估专家审查分析结果。

# 迭代优化

如果评估专家反馈分析结果未通过：
*   根据建议重新调用根因定位专家。
*   再次提交评估，直到结果被接受。

# 关键注意事项

*   **自主决策**：请优先使用日志相关的专家工具。
*   **时区**：本场景下，用户输入时间和日志时间均为 **UTC**。
"""

    def get_root_cause_localizer_prompt_template(self) -> str:
        return """你是根因定位器（Root Cause Localizer），负责基于日志进行故障分析和根因定位。

输入：用户提供的故障描述和时间范围。

任务：
1. 扫描关键日志文件（如 syslog, kernel, app logs）。
2. 识别错误模式、异常组件和故障发生时间。
3. 推断根本原因。

输出：JSON对象，root_causes数组包含 component、reason、fault_start_time（ISO, {timezone_info}）、logic_trace。

约束：本场景主要依赖日志。如果不确定，请如实说明。"""

class GenericLogDomainAdapter(DomainPromptAdapter):
    def get_prompt_hints(self) -> Dict[str, str]:
        return {
            "candidate_entities": "候选组件范围：基于日志目录下的所有日志文件",
            "core_metrics": "核心指标范围：本场景主要依赖日志内容分析（错误关键字、异常模式）。",
            "dataset_limitations": "数据集限制说明：仅包含日志数据，无指标和调用链数据。请充分利用日志中的时间戳和错误信息进行关联分析。",
            "timezone": "UTC"
        }

    def get_orchestrator_prompt(self) -> str:
        return """你是**根因分析协调器（RCA Orchestrator）**，负责协调通用日志场景下的故障诊断流程。

你的职责是协调领域专家完成诊断任务。你需要根据任务进展，自主决定调用哪个专家。

# 诊断任务目标

本场景为**通用日志分析模式**，主要依赖扫描和分析文件夹下的所有日志文件，请按以下流程推进：

1.  **日志异常分析**：调用根因定位专家（Root Cause Localizer），扫描所有可用的日志文件，识别错误模式、异常组件、故障发生时间，并直接推断根本原因。
2.  **评估与决策**：调用评估专家审查分析结果。

# 迭代优化

如果评估专家反馈分析结果未通过：
*   根据建议重新调用根因定位专家。
*   再次提交评估，直到结果被接受。

# 关键注意事项

*   **自主决策**：请优先使用日志相关的专家工具。
*   **灵活性**：日志文件名和内容格式可能多样，请根据实际读取到的内容进行分析。
*   **时区**：用户输入时间和日志时间默认为 **UTC**。
"""

    def get_root_cause_localizer_prompt_template(self) -> str:
        return """你是根因定位器（Root Cause Localizer），负责基于通用日志数据进行故障分析和根因定位。

输入：用户提供的故障描述和时间范围。

任务：
1. 扫描所有可用的日志源（Log Analysis Tool 会自动加载目录下所有日志）。
2. 识别日志中的错误模式（Error/Exception/Fail等）、异常组件（基于文件名区分）和故障发生时间。
3. 关联不同日志源中的异常信息，推断根本原因。

输出：JSON对象，root_causes数组包含 component、reason、fault_start_time（ISO, {timezone_info}）、logic_trace。

约束：
- 本场景主要依赖日志。
- 不要假设特定的日志文件名或组件名，根据实际数据分析。
- 如果不确定，请如实说明。"""

_DOMAIN_ADAPTER_REGISTRY: Dict[str, Callable[[], DomainPromptAdapter]] = {}

def register_domain_adapter(name: str, constructor: Callable[[], DomainPromptAdapter]) -> None:
    _DOMAIN_ADAPTER_REGISTRY[name.lower()] = constructor

def create_domain_adapter(config: Optional[Dict[str, Any]] = None) -> DomainPromptAdapter:
    cfg = config or {}
    name = str(cfg.get("domain_adapter", cfg.get("metric_adapter", "openrca"))).lower()
    if name not in _DOMAIN_ADAPTER_REGISTRY:
        return NullDomainAdapter()
    return _DOMAIN_ADAPTER_REGISTRY[name]()

register_domain_adapter("openrca", lambda: OpenRCADomainAdapter())
register_domain_adapter("disk_fault", lambda: DiskFaultDomainAdapter())
register_domain_adapter("generic_log", lambda: GenericLogDomainAdapter())
