"""
RCA Agent System

Implementation of DeepAgent-based root cause analysis system with specialized sub-agents.
"""

from typing import Optional, Dict, Any, List, Annotated
from typing_extensions import TypedDict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware, SummarizationMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.graph import BASE_AGENT_PROMPT
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from ..tools.log_tool import LogAnalysisTool
from ..tools.metric_tool import MetricAnalysisTool
from ..tools.trace_tool import TraceAnalysisTool
from .rca_config import (
    DEEP_AGENT_SYSTEM_PROMPT,
    ROOT_CAUSE_LOCALIZER_SYSTEM_PROMPT,
    get_metric_fault_analyst_prompt,
    get_evaluation_decision_prompt,
    get_evaluation_sub_agent_prompt,
    TOOL_CONFIGS
)
from .middleware import AgentHistoryRecordingMiddleware, AgentHistoryInjectionMiddleware, ExecutionHistory, SubAgentHistoryMergeMiddleware, AgentExecutionHistoryState


# Define state for evaluation decision agent graph
class EvaluationDecisionState(TypedDict, total=False):
    """State for evaluation decision agent graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    agent_execution_history: Dict[str, ExecutionHistory]
    evaluation_results: Annotated[List[BaseMessage], add_messages]
    final_decision: Optional[str]


def create_metric_fault_analyst_agent(
    model: BaseChatModel,
    config: Optional[Dict[str, Any]] = None
):
    tool_config = (config or {}).get("metric_analyzer") or TOOL_CONFIGS["metric_analyzer"]
    metric_tool = MetricAnalysisTool(config=tool_config)
    metric_tool.initialize()
    tools = metric_tool.get_tools()
    middleware = [
        TodoListMiddleware(),
        AgentHistoryRecordingMiddleware(record_agent_name="metric_fault_analyst"),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
        PatchToolCallsMiddleware(),
    ]
    return create_agent(
        model,
        tools=tools,
        system_prompt=(get_metric_fault_analyst_prompt(config) + "\n\n" + BASE_AGENT_PROMPT) if get_metric_fault_analyst_prompt(config) else BASE_AGENT_PROMPT,
        middleware=middleware,
    ).with_config({"recursion_limit": 1000})

def create_root_cause_localizer_agent(
    model: BaseChatModel,
    config: Optional[Dict[str, Any]] = None
):
    log_tool_config = (config or {}).get("log_analyzer") or TOOL_CONFIGS["log_analyzer"]
    trace_tool_config = (config or {}).get("trace_analyzer") or TOOL_CONFIGS["trace_analyzer"]
    log_tool = LogAnalysisTool(config=log_tool_config)
    trace_tool = TraceAnalysisTool(config=trace_tool_config)
    log_tool.initialize()
    trace_tool.initialize()
    tools = log_tool.get_tools() + trace_tool.get_tools()
    middleware = [
        TodoListMiddleware(),
        AgentHistoryRecordingMiddleware(record_agent_name="root_cause_localizer"),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
        PatchToolCallsMiddleware(),
    ]
    return create_agent(
        model,
        tools=tools,
        system_prompt=ROOT_CAUSE_LOCALIZER_SYSTEM_PROMPT + "\n\n" + BASE_AGENT_PROMPT if ROOT_CAUSE_LOCALIZER_SYSTEM_PROMPT else BASE_AGENT_PROMPT,
        middleware=middleware,
    ).with_config({"recursion_limit": 1000})

def create_evaluation_sub_agent(
    model: BaseChatModel,
    agent_name: str,
    config: Optional[Dict[str, Any]] = None
):
    """
    Create the evaluation sub-agent with history injection middleware.
    
    Args:
        model: Language model to use for the agent
        agent_name: Name of the evaluation agent (e.g., "evaluation_agent_1")
        config: Configuration dictionary
        
    Returns:
        Agent graph for evaluation with history injection middleware
    """
    middleware = [
        AgentHistoryInjectionMiddleware(),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
    ]
    
    agent_graph = create_agent(
        model=model,
        tools=[],
        system_prompt=get_evaluation_sub_agent_prompt(config),
        middleware=middleware
    )
    return agent_graph


def create_evaluation_decision_agent(
    model: BaseChatModel,
    config: Optional[Dict[str, Any]] = None,
    num_evaluation_agents: int = 3
):
    """
    Create the evaluation decision agent using LangGraph with parallel evaluation nodes.
    
    Args:
        model: Language model to use for the agent
        config: Configuration dictionary
        num_evaluation_agents: Number of parallel evaluation agents (default: 3)
        
    Returns:
        Compiled LangGraph for evaluation decision
    """
    # Create evaluation sub-agents using loop
    evaluation_agents = {}
    evaluation_agent_names = []
    
    for i in range(1, num_evaluation_agents + 1):
        agent_name = f"evaluation_agent_{i}"
        evaluation_agents[agent_name] = create_evaluation_sub_agent(model, agent_name, config)
        evaluation_agent_names.append(agent_name)
    
    # Create decision agent
    decision_agent = create_agent(
                    model=model,
        tools=[],
        system_prompt=get_evaluation_decision_prompt(config)
    )
    
    def create_evaluation_node(agent, agent_name: str):
        """Create an evaluation node function."""
        def evaluation_node(state: EvaluationDecisionState):
            """Evaluation node that runs an evaluation agent.
            
            History injection is handled by AgentHistoryInjectionMiddleware in the agent.
            """
            # Set agent_name in state so middleware can identify it
            state_with_name = {**state, "agent_name": agent_name}
            
            # Run the evaluation agent directly with state
            # Middleware will inject history automatically based on agent_name
            from langchain_core.runnables import RunnableConfig
            config = RunnableConfig(configurable={"agent_name": agent_name})
            result = agent.invoke(state_with_name, config=config)
            
            # Extract the evaluation result message (last message)
            evaluation_result_message = None
            if result.get("messages"):
                evaluation_result_message = result["messages"][-1]
            
            # Update state with evaluation result message
            if evaluation_result_message:
                return {
                    "evaluation_results": evaluation_result_message
                }
            return {}
        
        return evaluation_node
    
    def decision_node(state: EvaluationDecisionState):
        """Decision node that makes final decision based on evaluation results."""
        # Collect evaluation results from list
        eval_results_messages = state.get("evaluation_results", [])
        
        # Prepare input for decision agent
        eval_results_text = []
        for i, msg in enumerate(eval_results_messages, 1):
            agent_name = evaluation_agent_names[i - 1] if i <= len(evaluation_agent_names) else f"evaluation_agent_{i}"
            result_content = msg.content if hasattr(msg, "content") else str(msg)
            eval_results_text.append(f"## {agent_name}的结果：\n{result_content or '未完成评估'}")
        
        # Join results with newline (cannot use "\n" directly in f-string expression)
        eval_results_joined = "\n".join(eval_results_text)
        
        decision_input = f"""请基于以下评估agent的评估结果进行综合分析并做出最终决策：

{eval_results_joined}

请综合分析这些评估结果，做出最终决策。"""
        
        from langchain_core.messages import HumanMessage
        decision_messages = [HumanMessage(content=decision_input)]
        
        # Run decision agent directly with state
        decision_state = {
            **state,
            "messages": decision_messages,
        }
        
        result = decision_agent.invoke(decision_state)
        
        # Extract the decision result
        final_decision = None
        if result.get("messages"):
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                final_decision = last_message.content
        
        return {
            "final_decision": final_decision,
            "messages": result.get("messages", [])
        }
    
    # Build the graph
    graph = StateGraph(EvaluationDecisionState)
    
    # Create and add evaluation nodes using loop
    for agent_name in evaluation_agent_names:
        evaluation_node = create_evaluation_node(evaluation_agents[agent_name], agent_name)
        graph.add_node(agent_name, evaluation_node)
    
    # Add decision node
    graph.add_node("decision_agent", decision_node)
    
    # Add edges: START -> parallel evaluation nodes
    for agent_name in evaluation_agent_names:
        graph.add_edge(START, agent_name)
    
    # Add edge: all evaluation nodes -> decision node (all must complete)
    graph.add_edge(evaluation_agent_names, "decision_agent")
    
    # Add edge: decision node -> END
    graph.add_edge("decision_agent", END)
    
    # Compile and return the graph
    return graph.compile()


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
    
    # 1. 创建合并后的 Metric Fault Analyst
    metric_fault_analyst_agent = create_metric_fault_analyst_agent(sub_agent_model, config)
    
    # 2. 创建 Root Cause Localizer (保持不变)
    root_cause_localizer_agent = create_root_cause_localizer_agent(sub_agent_model, config)
    
    # 3. 创建评估决策 Agent
    evaluation_decision_agent = create_evaluation_decision_agent(sub_agent_model, config)

    # 4. 封装 SubAgents
    subagents = [
        CompiledSubAgent(
            name="metric_fault_analysis_agent",
            # 描述合并了 Step 1 和 Step 2
            description="Step 1 Agent: Responsible for Metric Analysis. It handles data preprocessing, threshold calculation, anomaly detection, AND noise filtering to identify confirmed faulty components.",
            runnable=metric_fault_analyst_agent
        ),
        CompiledSubAgent(
            name="root_cause_localization_agent",
            # 现在的 Step 2
            description="Step 2 Agent: Responsible for Root Cause Localization using Traces and Logs based on the confirmed faults.",
            runnable=root_cause_localizer_agent
        ),
        CompiledSubAgent(
            name="evaluation_decision_agent",
            # Step 3
            description="Step 3 Agent: Responsible for Evaluation and Decision. Evaluates the analysis results from previous agents and makes final decisions based on execution history.",
            runnable=evaluation_decision_agent
        ),
    ]

    middleware = [
        TodoListMiddleware(),
        SubAgentHistoryMergeMiddleware(),
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
        model=model,
        name="rca_agent",
        system_prompt=DEEP_AGENT_SYSTEM_PROMPT + "\n\n" + BASE_AGENT_PROMPT if DEEP_AGENT_SYSTEM_PROMPT else BASE_AGENT_PROMPT,
        middleware=middleware,
    ).with_config({"recursion_limit": 1000})
