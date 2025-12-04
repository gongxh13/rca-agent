"""
RCA Agent System

Implementation of DeepAgent-based root cause analysis system with specialized sub-agents.
"""

from typing import Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent
from deepagents import create_deep_agent, CompiledSubAgent

from ..tools.local_log_tool import LocalLogAnalysisTool
from ..tools.local_metric_tool import LocalMetricAnalysisTool
from ..tools.local_trace_tool import LocalTraceAnalysisTool
from .rca_config import (
    DEEP_AGENT_SYSTEM_PROMPT,
    LOG_AGENT_PROMPT,
    METRIC_AGENT_PROMPT,
    TRACE_AGENT_PROMPT,
    TOOL_CONFIGS
)


def create_log_analysis_agent(
    model: BaseChatModel,
    config: Optional[Dict[str, Any]] = None
):
    """
    Create a specialized log analysis agent.
    
    Args:
        model: Language model to use for the agent
        config: Configuration for the log analysis tool
        
    Returns:
        Agent graph for log analysis
    """
    # Initialize log analysis tool
    tool_config = config or TOOL_CONFIGS["log_analyzer"]
    log_tool = LocalLogAnalysisTool(config=tool_config)
    log_tool.initialize()
    
    # Get tools from the tool class
    tools = log_tool.get_tools()

    tools = []
    
    # Add PythonREPLTool as fallback (LAST RESORT)
    from langchain_experimental.tools import PythonREPLTool
    python_repl = PythonREPLTool()
    tools.append(python_repl)
    
    # Create agent using create_agent
    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=LOG_AGENT_PROMPT
    )
    
    return agent_graph


def create_metric_analysis_agent(
    model: BaseChatModel,
    config: Optional[Dict[str, Any]] = None
):
    """
    Create a specialized metric analysis agent.
    
    Args:
        model: Language model to use for the agent
        config: Configuration for the metric analysis tool
        
    Returns:
        Agent graph for metric analysis
    """
    # Initialize metric analysis tool
    tool_config = config or TOOL_CONFIGS["metric_analyzer"]
    metric_tool = LocalMetricAnalysisTool(config=tool_config)
    metric_tool.initialize()
    
    # Get tools from the tool class
    tools = metric_tool.get_tools()

    tools = []
    # Add PythonREPLTool as fallback (LAST RESORT)
    from langchain_experimental.tools import PythonREPLTool
    python_repl = PythonREPLTool()
    tools.append(python_repl)
    
    # Create agent using create_agent
    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=METRIC_AGENT_PROMPT
    )
    
    return agent_graph


def create_trace_analysis_agent(
    model: BaseChatModel,
    config: Optional[Dict[str, Any]] = None
):
    """
    Create a specialized trace analysis agent.
    
    Args:
        model: Language model to use for the agent
        config: Configuration for the trace analysis tool
        
    Returns:
        Agent graph for trace analysis
    """
    # Initialize trace analysis tool
    tool_config = config or TOOL_CONFIGS["trace_analyzer"]
    trace_tool = LocalTraceAnalysisTool(config=tool_config)
    trace_tool.initialize()
    
    # Get tools from the tool class
    tools = trace_tool.get_tools()

    tools = []
    
    # Add PythonREPLTool as fallback (LAST RESORT)
    from langchain_experimental.tools import PythonREPLTool
    python_repl = PythonREPLTool()
    tools.append(python_repl)
    
    # Create agent using create_agent
    agent_graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=TRACE_AGENT_PROMPT
    )
    
    return agent_graph


def create_rca_deep_agent(
    model: BaseChatModel,
    sub_agent_model: Optional[BaseChatModel] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Create the DeepAgent coordinator for root cause analysis.
    
    Args:
        model: Language model to use for the DeepAgent coordinator
        sub_agent_model: Language model to use for sub-agents (defaults to same as coordinator)
        config: Configuration dictionary for tools
        
    Returns:
        DeepAgent instance configured for RCA
    """
    # Use same model for sub-agents if not specified
    if sub_agent_model is None:
        sub_agent_model = model
    
    # Create sub-agents
    log_agent = create_log_analysis_agent(sub_agent_model, config)
    metric_agent = create_metric_analysis_agent(sub_agent_model, config)
    trace_agent = create_trace_analysis_agent(sub_agent_model, config)
    
    # Wrap sub-agents as CompiledSubAgents
    subagents = [
        CompiledSubAgent(
            name="log-analyzer",
            description="Specialized agent for analyzing application and system logs. Can find error patterns, detect anomalies, analyze error frequencies, and correlate log events.",
            runnable=log_agent
        ),
        CompiledSubAgent(
            name="metric-analyzer",
            description="Specialized agent for analyzing application performance and infrastructure metrics. Can analyze service performance, resource usage, detect metric anomalies, and assess component health.",
            runnable=metric_agent
        ),
        CompiledSubAgent(
            name="trace-analyzer",
            description="Specialized agent for analyzing distributed traces. Can find slow spans, analyze call chains, map service dependencies, detect latency anomalies, and identify bottlenecks.",
            runnable=trace_agent
        ),
    ]
    
    # Create DeepAgent
    deep_agent = create_deep_agent(
        model=model,
        tools=[],  # DeepAgent doesn't use tools directly, only sub-agents
        system_prompt=DEEP_AGENT_SYSTEM_PROMPT,
        subagents=subagents
    )
    
    return deep_agent
