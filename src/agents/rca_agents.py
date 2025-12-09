"""
RCA Agent System

Implementation of DeepAgent-based root cause analysis system with specialized sub-agents.
"""

from typing import Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware, SummarizationMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.graph import BASE_AGENT_PROMPT
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

from ..tools.local_log_tool import LocalLogAnalysisTool
from ..tools.local_metric_tool import LocalMetricAnalysisTool
from ..tools.local_trace_tool import LocalTraceAnalysisTool
from .rca_config import (
    DEEP_AGENT_SYSTEM_PROMPT,
    LOG_AGENT_PROMPT,
    METRIC_AGENT_PROMPT,
    TRACE_AGENT_PROMPT,
    DECISION_REFLECTION_AGENT_PROMPT,
    METRIC_FAULT_ANALYST_AGENT_SYSTEM_PROMPT,
    ROOT_CAUSE_LOCALIZER_SYSTEM_PROMPT,
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


def create_decision_reflection_agent(
    model: BaseChatModel,
    config: Optional[Dict[str, Any]] = None
):
    agent_graph = create_agent(
        model=model,
        tools=[],
        system_prompt=DECISION_REFLECTION_AGENT_PROMPT
    )
    return agent_graph


def create_metric_fault_analyst_agent(
    model: BaseChatModel,
    config: Optional[Dict[str, Any]] = None
):
    metric_agent = create_metric_analysis_agent(model, config)
    subagents = [
        CompiledSubAgent(
            name="metric-analyzer",
            description="Specialized agent for analyzing application performance and infrastructure metrics. Can analyze service performance, resource usage, detect metric anomalies, and assess component health.",
            runnable=metric_agent
        ),
    ]

    middleware = [
        TodoListMiddleware(),
        SubAgentMiddleware(
            default_model=model,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(),
                SummarizationMiddleware(
                    model=model,
                    max_tokens_before_summary=170000,
                    messages_to_keep=6,
                ),
                PatchToolCallsMiddleware(),
            ],
            general_purpose_agent=True,
        ),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
        PatchToolCallsMiddleware(),
    ]

    return create_agent(
        model,
        system_prompt=METRIC_FAULT_ANALYST_AGENT_SYSTEM_PROMPT + "\n\n" + BASE_AGENT_PROMPT if METRIC_FAULT_ANALYST_AGENT_SYSTEM_PROMPT else BASE_AGENT_PROMPT,
        middleware=middleware,
    ).with_config({"recursion_limit": 1000})
    # # Create DeepAgent
    # deep_agent = create_deep_agent(
    #     model=model,
    #     system_prompt=METRIC_FAULT_ANALYST_AGENT_SYSTEM_PROMPT,
    #     subagents=subagents
    # )
    # return deep_agent

def create_root_cause_localizer_agent(
    model: BaseChatModel,
    config: Optional[Dict[str, Any]] = None
):
    log_agent = create_log_analysis_agent(model, config)
    trace_agent = create_trace_analysis_agent(model, config)
    subagents = [
        CompiledSubAgent(
            name="log-analyzer",
            description="Specialized agent for analyzing application and system logs. Can find error patterns, detect anomalies, analyze error frequencies, and correlate log events.",
            runnable=log_agent
        ),
        CompiledSubAgent(
            name="trace-analyzer",
            description="Specialized agent for analyzing distributed traces. Can find slow spans, analyze call chains, map service dependencies, detect latency anomalies, and identify bottlenecks.",
            runnable=trace_agent
        ),
    ]

    middleware = [
        TodoListMiddleware(),
        SubAgentMiddleware(
            default_model=model,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(),
                SummarizationMiddleware(
                    model=model,
                    max_tokens_before_summary=170000,
                    messages_to_keep=6,
                ),
                PatchToolCallsMiddleware(),
            ],
            general_purpose_agent=True,
        ),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
        PatchToolCallsMiddleware(),
    ]

    return create_agent(
        model,
        system_prompt=ROOT_CAUSE_LOCALIZER_SYSTEM_PROMPT + "\n\n" + BASE_AGENT_PROMPT if METRIC_FAULT_ANALYST_AGENT_SYSTEM_PROMPT else BASE_AGENT_PROMPT,
        middleware=middleware,
    ).with_config({"recursion_limit": 1000})
    # Create DeepAgent
    # deep_agent = create_deep_agent(
    #     model=model,
    #     system_prompt=ROOT_CAUSE_LOCALIZER_SYSTEM_PROMPT,
    #     subagents=subagents
    # )
    # return deep_agent

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
    decision_agent = create_decision_reflection_agent(sub_agent_model, config)

    # 1. 创建合并后的 Metric Fault Analyst
    metric_fault_analyst_agent = create_metric_fault_analyst_agent(sub_agent_model, config)
    
    # 2. 创建 Root Cause Localizer (保持不变)
    root_cause_localizer_agent = create_root_cause_localizer_agent(sub_agent_model, config)

    # 3. 封装 SubAgents
    subagents = [
        CompiledSubAgent(
            name="metric_fault_analyst",
            # 描述合并了 Step 1 和 Step 2
            description="Step 1 Agent: Responsible for Metric Analysis. It handles data preprocessing, threshold calculation, anomaly detection, AND noise filtering to identify confirmed faulty components.",
            runnable=metric_fault_analyst_agent
        ),
        CompiledSubAgent(
            name="root_cause_localizer",
            # 现在的 Step 2
            description="Step 2 Agent: Responsible for Root Cause Localization using Traces and Logs based on the confirmed faults.",
            runnable=root_cause_localizer_agent
        ),
    ]

    middleware = [
        TodoListMiddleware(),
        SubAgentMiddleware(
            default_model=model,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(),
                SummarizationMiddleware(
                    model=model,
                    max_tokens_before_summary=170000,
                    messages_to_keep=6,
                ),
                PatchToolCallsMiddleware(),
            ],
            general_purpose_agent=True,
        ),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
        PatchToolCallsMiddleware(),
    ]

    return create_agent(
        model,
        system_prompt=DEEP_AGENT_SYSTEM_PROMPT + "\n\n" + BASE_AGENT_PROMPT if DEEP_AGENT_SYSTEM_PROMPT else BASE_AGENT_PROMPT,
        middleware=middleware,
    ).with_config({"recursion_limit": 1000})
    
    # # Create DeepAgent
    # deep_agent = create_deep_agent(
    #     model=model,
    #     tools=[],  # DeepAgent doesn't use tools directly, only sub-agents
    #     system_prompt=DEEP_AGENT_SYSTEM_PROMPT,
    #     subagents=subagents
    # )
    
    # return deep_agent
