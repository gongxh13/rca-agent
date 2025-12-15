"""
Flamegraph Analysis Agent

Implementation of agent for analyzing CPU flamegraph files to identify performance bottlenecks.
"""

from typing import Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent

from .flamegraph_config import FLAMEGRAPH_ANALYSIS_AGENT_SYSTEM_PROMPT


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
    
    # Create agent using create_agent
    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=FLAMEGRAPH_ANALYSIS_AGENT_SYSTEM_PROMPT
    )
    
    return agent_graph

