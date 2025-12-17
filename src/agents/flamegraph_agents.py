"""
Flamegraph Analysis Agent

Implementation of agent for analyzing CPU flamegraph files to identify performance bottlenecks.
"""

from typing import Optional, Dict, Any, Callable
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent

from .flamegraph_config import (
    FLAMEGRAPH_ANALYSIS_AGENT_SYSTEM_PROMPT,
    FLAMEGRAPH_AUTO_PROFILING_AGENT_SYSTEM_PROMPT,
)


def create_flamegraph_analysis_agent(
    model: BaseChatModel,
    config: Optional[Dict[str, Any]] = None
):
    """
    Create a flamegraph analysis agent for CPU performance analysis.
    
    Args:
        model: Language model to use for the agent
        config: Configuration dictionary (optional, not used currently)
        
    Returns:
        Agent graph for flamegraph analysis
    """
    # Import flamegraph tools (only analysis tools, not profiling tools)
    from ..tools.flamegraph_cpu_analyzer import (
        flamegraph_overview,
        flamegraph_drill_down
    )
    
    # Get tools
    tools = [
        flamegraph_overview,
        flamegraph_drill_down,
    ]
    
    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=FLAMEGRAPH_ANALYSIS_AGENT_SYSTEM_PROMPT
    )
    
    return agent_graph


def create_flamegraph_auto_profiling_agent(
    model: BaseChatModel,
    config: Optional[Dict[str, Any]] = None
):
    """
    Create a flamegraph agent that can auto-collect (profiling) and then analyze.

    This agent exposes:
    - profiling tools (start/stop/list) for flamegraph collection
    - analysis tools (overview/drill_down) for interpretation
    """
    from ..tools.flamegraph_cpu_analyzer import (
        flamegraph_overview,
        flamegraph_drill_down,
        FlamegraphProfilingTool,
    )

    profiling_tool = FlamegraphProfilingTool()
    profiling_tools = profiling_tool.get_tools()

    tools = [
        *profiling_tools,
        flamegraph_overview,
        flamegraph_drill_down,
    ]

    # Prefer ShellToolMiddleware (if available) so the model can inspect processes and pick PID itself.
    middleware = []
    try:  # pragma: no cover
        from langchain.agents.middleware import ShellToolMiddleware  # type: ignore
        middleware.append(ShellToolMiddleware())
    except Exception:
        middleware = []

    # Some create_agent implementations may not accept middleware; fall back gracefully.
    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=FLAMEGRAPH_AUTO_PROFILING_AGENT_SYSTEM_PROMPT,
        middleware=middleware if middleware else None,
    )

    return agent_graph

