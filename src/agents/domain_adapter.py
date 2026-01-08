from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

class DomainPromptAdapter(ABC):
    @abstractmethod
    def get_prompt_hints(self) -> Dict[str, str]:
        ...

class NullDomainAdapter(DomainPromptAdapter):
    def get_prompt_hints(self) -> Dict[str, str]:
        return {
            "candidate_entities": "",
            "core_metrics": "",
            "dataset_limitations": ""
        }

class OpenRCADomainAdapter(DomainPromptAdapter):
    def get_prompt_hints(self) -> Dict[str, str]:
        return {
            "candidate_entities": "候选组件范围：仅关注 apache01、apache02、Tomcat01-04、Mysql01-02、Redis01-02、MG01-02、IG01-02。",
            "core_metrics": "核心指标范围：CPU、内存、磁盘I/O、磁盘空间、JVM CPU Load、JVM OOM（HeapMemoryUsed/HeapMemoryMax）、网络（带宽、错误、TCP连接、容器Rx/Tx）。",
            "dataset_limitations": "数据集限制说明：OpenRCA 数据集可能不包含某些系统级指标（如共享存储、节点级资源、部分基础设施或网络存储设备性能），评估时需区分数据缺失与分析遗漏，仅基于实际可用数据进行评估与建议。"
        }

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
