from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, List, Union

class DomainPromptAdapter(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

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
            "candidate_entities": "候选组件范围：仅关注 apache01、apache02、Tomcat01, Tomcat02, Tomcat03, Tomcat04、Mysql01, Mysql02、Redis01, Redis02、MG01, MG02、IG01, IG02。",
            "core_metrics": "核心指标范围：CPU、内存、磁盘I/O (DSKRead/Write)、磁盘空间、JVM（GC、Heap/NoHeap、ThreadCount）、数据库（连接数、锁等待、缓冲池）、网络（流量 Packets/KB、TCP状态 CLOSE-WAIT/FIN-WAIT）、应用层（响应时间 MRT、错误计数/SR）。",
            "dataset_limitations": "数据集限制说明：1. OpenRCA 数据集可能不包含直接的网络延迟（Ping）或丢包率指标，请结合 TCP 连接状态、网络包量变化及应用层响应时间/错误日志进行推断。2. 应用层指标（ServiceTest）反映整体健康度，但可能无法直接关联调用链（Trace），定位时请主要依赖组件级指标（Tomcat/MySQL等）。",
            "timezone": "UTC+8"
        }

    def get_orchestrator_prompt(self) -> str:
        # Check if evaluation is enabled (default True)
        enable_evaluation = self.config.get("evaluation", {}).get("enable", True)

        prompt = """你是**根因分析协调器（RCA Orchestrator）**，负责协调分布式系统故障诊断的完整流程。

你的职责是协调多个领域专家完成诊断任务。你需要根据任务进展，自主决定调用哪个专家，并在专家之间传递上下文信息。

# 诊断任务目标

利用可用的专家工具，完成以下诊断阶段：

1.  **异常检测与定界**：调用相关专家分析故障时间窗口内的关键指标，识别异常组件和指标偏离。
2.  **根因定位**：基于检测到的异常组件，调用专家结合调用链（Trace）和日志（Log）进行深入分析，定位根本原因。"""

        if enable_evaluation:
            prompt += """
3.  **评估与决策**：调用评估专家审查分析结果的合理性、完整性和准确性。

# 迭代优化

如果评估专家反馈分析结果未通过（`reject` 或 `need_more_analysis`）：
*   请仔细阅读其提供的改进建议。
*   重新调用相应的分析专家进行补充分析。
*   再次提交评估，直到结果被接受。"""
        else:
            prompt += """
3.  **最终输出**：直接输出最终的根因分析报告。"""

        prompt += """

# 关键注意事项

*   **故障数量一致性（重要）**：
    *   分析用户输入的问题描述，确定其中包含的独立故障数量。
    *   如果用户描述了**单次/单个**故障，最终输出必须聚焦于**一个**核心根因。
    *   如果用户描述了**多次/多个**故障（例如“上午发生一次，下午又发生一次”或“A服务和B服务分别报错”），最终输出应包含**对应数量**的根因。
    *   请在调用专家工具时，明确传达这一数量约束。

*   **自主决策**：根据任务描述选择最匹配的专家工具（Expert Tool）。例如，涉及指标分析时选择指标专家，涉及根因推理时选择根因定位专家。
*   **上下文传递**：调用下一个专家时，务必将上一个专家的关键输出作为输入传递。
*   **时区**：用户输入时间默认为 UTC+8。
"""
        return prompt

    def get_root_cause_localizer_prompt_template(self) -> str:
        return """你是根因定位器（Root Cause Localizer），负责在异常列表基础上确定最终根因。

输入：指标故障分析师输出的已确认故障组件列表和用户原始描述。

根因数量确定（极重要）：
1. 仔细分析用户原始描述中提到的故障场景数量。
2. 如果用户只描述了一个故障（例如：“服务响应变慢”），你必须只输出一个最可能的根因。
3. 如果用户明确描述了多个故障（例如：“上午10点报错一次，下午3点又报错一次”），你必须分别为每一次故障定位根因，并输出对应数量的结果。
4. 根因数量应与用户描述的故障发生次数或独立故障场景完全匹配。

输出：JSON对象，root_causes数组包含 component、reason、fault_start_time（ISO, {timezone_info}）、logic_trace。

约束：不编造，当日志/调用链无证据时如实说明未知；仅使用已提供工具完成分析。"""

class DiskFaultDomainAdapter(DomainPromptAdapter):
    def get_prompt_hints(self) -> Dict[str, str]:
        return {
            "candidate_entities": "候选组件范围：主要关注 kernel, syslog, app 三个日志源，以及可能的磁盘设备（如 scsi_debug, sda, vda 等）。",
            "core_metrics": "核心指标范围：本场景主要依赖日志分析",
            "dataset_limitations": "数据集限制说明：目前仅包含 app/kernel/syslog 日志，无详细指标数据。需通过日志模式匹配",
            "timezone": "Asia/Shanghai"
        }

    def get_orchestrator_prompt(self) -> str:
        # Check if evaluation is enabled (default True)
        enable_evaluation = self.config.get("evaluation", {}).get("enable", True)

        prompt = """你是**根因分析协调器（RCA Orchestrator）**，负责协调磁盘故障诊断的完整流程。

你的职责是协调领域专家完成诊断任务。你需要根据任务进展，自主决定调用哪个专家。

# 核心职责：时间窗口确定

在调用专家工具前，你必须明确故障分析的时间窗口。请根据用户输入和上下文中的当前系统时间（Current system time）进行判断：

1.  **显式时间窗口**：
    *   若用户提供了具体时间范围（如“10:00到10:30”），直接使用该范围。
    *   请结合日期信息，确保生成完整的时间戳。

2.  **隐式/模糊时间窗口**：
    *   若用户仅提到“刚刚”、“最近”或未提及时间，需根据当前时间自动推断。
    *   **策略**：默认截取过去 **10分钟** 的窗口（也可根据语境选择5或1小时）。
    *   例如：当前时间为 2026-01-23 10:00:00，用户说“刚刚”，则窗口定为 09:50:00 至 10:00:00。

# 诊断任务目标

本场景主要依赖日志分析，请按以下流程推进：

1.  **日志异常分析**：
    *   调用根因定位专家（Root Cause Localizer）。
    *   **必须在任务描述中包含明确的时间窗口**（Start Time, End Time）。
    *   任务示例：“分析 2026-01-23 09:30:00 到 10:00:00 期间的日志，识别异常...”。"""

        if enable_evaluation:
            prompt += """
2.  **评估与决策**：调用评估专家审查分析结果。

# 迭代优化

如果评估专家反馈分析结果未通过：
*   根据建议重新调用根因定位专家。
*   再次提交评估，直到结果被接受。"""
        else:
            prompt += """
2.  **最终输出**：直接输出最终的根因分析报告。"""

        prompt += """

# 关键注意事项

*   **自主决策**：请优先使用日志相关的专家工具。
*   **时区**：本场景默认为 **Asia/Shanghai (UTC+8)**，请确保时间处理一致。
*   **明确指令**：子Agent（根因定位器）需要精确的时间范围来过滤日志，不要下发模糊的时间指令。
"""
        return prompt

    def get_root_cause_localizer_prompt_template(self) -> str:
        return """你是根因定位器（Root Cause Localizer），负责基于日志进行故障分析和根因定位。

输入：用户提供的故障描述和时间范围。

任务：
1. 扫描关键日志文件（如 syslog, kernel, app logs）。
2. 识别错误模式、异常组件和故障发生时间。
3. 推断根本原因。如果在某个窗口检查到多个故障，以最近的一次为主。

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
        # Check if evaluation is enabled (default True)
        enable_evaluation = self.config.get("evaluation", {}).get("enable", True)

        prompt = """你是**根因分析协调器（RCA Orchestrator）**，负责协调通用日志场景下的故障诊断流程。

你的职责是协调领域专家完成诊断任务。你需要根据任务进展，自主决定调用哪个专家。

# 诊断任务目标

本场景为**通用日志分析模式**，主要依赖扫描和分析文件夹下的所有日志文件，请按以下流程推进：

1.  **日志异常分析**：调用根因定位专家（Root Cause Localizer），扫描所有可用的日志文件，识别错误模式、异常组件、故障发生时间，并直接推断根本原因。"""

        if enable_evaluation:
            prompt += """
2.  **评估与决策**：调用评估专家审查分析结果。

# 迭代优化

如果评估专家反馈分析结果未通过：
*   根据建议重新调用根因定位专家。
*   再次提交评估，直到结果被接受。"""
        else:
            prompt += """
2.  **最终输出**：直接输出最终的根因分析报告。"""

        prompt += """

# 关键注意事项

*   **自主决策**：请优先使用日志相关的专家工具。
*   **灵活性**：日志文件名和内容格式可能多样，请根据实际读取到的内容进行分析。
*   **时区**：用户输入时间和日志时间默认为 **UTC**。
"""
        return prompt

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

_DOMAIN_ADAPTER_REGISTRY: Dict[str, Callable[[Optional[Dict[str, Any]]], DomainPromptAdapter]] = {}

def register_domain_adapter(name: str, constructor: Callable[[Optional[Dict[str, Any]]], DomainPromptAdapter]) -> None:
    _DOMAIN_ADAPTER_REGISTRY[name.lower()] = constructor

def create_domain_adapter(config: Optional[Dict[str, Any]] = None) -> DomainPromptAdapter:
    cfg = config or {}
    name = str(cfg.get("domain_adapter", cfg.get("metric_adapter", "openrca"))).lower()
    if name not in _DOMAIN_ADAPTER_REGISTRY:
        return NullDomainAdapter()
    return _DOMAIN_ADAPTER_REGISTRY[name](config)

register_domain_adapter("openrca", lambda config=None: OpenRCADomainAdapter(config))
register_domain_adapter("disk_fault", lambda config=None: DiskFaultDomainAdapter(config))
register_domain_adapter("generic_log", lambda config=None: GenericLogDomainAdapter(config))
