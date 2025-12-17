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

# RCA agents are optional at import-time (avoid breaking flamegraph-only usage)
try:  # pragma: no cover
    from .rca_agents import (
        create_log_analysis_agent,
        create_metric_analysis_agent,
        create_trace_analysis_agent,
        create_rca_deep_agent,
    )
except Exception:  # pragma: no cover
    create_log_analysis_agent = None  # type: ignore
    create_metric_analysis_agent = None  # type: ignore
    create_trace_analysis_agent = None  # type: ignore
    create_rca_deep_agent = None  # type: ignore

__all__ = [
    "create_flamegraph_analysis_agent",
    "create_flamegraph_auto_profiling_agent",
    "create_log_analysis_agent",
    "create_metric_analysis_agent",
    "create_trace_analysis_agent",
    "create_rca_deep_agent",
]
