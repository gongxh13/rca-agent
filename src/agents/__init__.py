"""
Agents Module

Contains agent implementations for root cause analysis and flamegraph analysis.

Note:
- Some environments may not have all optional dependencies / APIs (e.g. different
  LangChain versions). We keep imports resilient so flamegraph features can be
  used independently from RCA features.
"""

from .flamegraph_agents import (
    create_flamegraph_analysis_agent,
    create_flamegraph_auto_profiling_agent,
)

from .rca_agents import (
    create_sub_agent_middleware,
)

try:
    from src.agents.agent_state import agent_registry
    agent_registry.register("rca", create_sub_agent_middleware())
except ImportError:
    print("Failed to import agent_registry")
    pass

__all__ = [
    "create_flamegraph_analysis_agent",
    "create_flamegraph_auto_profiling_agent",
]
