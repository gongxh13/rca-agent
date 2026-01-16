"""
Custom Middleware for RCA Agent System

This module provides middleware for recording and injecting agent execution history.
"""

from typing import Annotated, Dict, List, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware import AgentMiddleware


def merge_dicts(current_state: dict, new_value: dict) -> dict:
    current_state.update(new_value)
    return current_state

# Define execution history structure for a single agent
class ExecutionHistory(TypedDict):
    """Execution history for a single agent."""
    messages: List[BaseMessage]


def create_empty_execution_history() -> ExecutionHistory:
    """Create an empty execution history."""
    return ExecutionHistory(messages=[])


# Define the state schema - only contains the execution history map
# Other fields (messages, input, output, etc.) come from langchain's standard state
class AgentExecutionHistoryState(TypedDict, total=False):
    """State schema that extends langchain's standard state with agent execution history tracking."""
    extend_data: Annotated[Dict[str, Any], merge_dicts]


class AgentHistoryRecordingMiddleware(AgentMiddleware):
    """
    Middleware that records execution history for the current agent.
    
    This middleware captures the execution history and stores it under a specified key.
    Uses after_agent hook to capture the full execution context.
    """
    
    # Define state schema for this middleware
    state_schema = AgentExecutionHistoryState
    
    def __init__(self, record_agent_name: str):
        """
        Initialize the middleware.
        
        Args:
            record_agent_name: Name to use as key when saving execution history.
        """
        super().__init__()
        self.record_agent_name = record_agent_name
    
    def after_agent(self, state: AgentExecutionHistoryState, config: RunnableConfig) -> AgentExecutionHistoryState:
        """
        Called after agent execution. Store the messages history.
        """
        # Initialize extend_data if not exists
        if "extend_data" not in state:
            state["extend_data"] = {}
            
        # Initialize history storage if not exists
        if "agent_execution_history" not in state["extend_data"]:
            state["extend_data"]["agent_execution_history"] = {}
        
        # Initialize agent history if not exists
        if self.record_agent_name not in state["extend_data"]["agent_execution_history"]:
            state["extend_data"]["agent_execution_history"][self.record_agent_name] = create_empty_execution_history()
        
        # Store messages from current state
        current_messages = state.get("messages", [])
        existing_messages = state["extend_data"]["agent_execution_history"][self.record_agent_name]["messages"]
        # Only add new messages that aren't already in the history
        existing_message_ids = {id(msg) for msg in existing_messages}
        new_messages = [msg for msg in current_messages if id(msg) not in existing_message_ids]
        state["extend_data"]["agent_execution_history"][self.record_agent_name]["messages"].extend(new_messages)
        
        return state
    
    async def aafter_agent(self, state: AgentExecutionHistoryState, config: RunnableConfig) -> AgentExecutionHistoryState:
        """
        Async version of after_agent. Store the messages history.
        """
        # Initialize extend_data if not exists
        if "extend_data" not in state:
            state["extend_data"] = {}

        # Initialize history storage if not exists
        if "agent_execution_history" not in state["extend_data"]:
            state["extend_data"]["agent_execution_history"] = {}
        
        # Initialize agent history if not exists
        if self.record_agent_name not in state["extend_data"]["agent_execution_history"]:
            state["extend_data"]["agent_execution_history"][self.record_agent_name] = create_empty_execution_history()
        
        # Store messages from current state
        current_messages = state.get("messages", [])
        existing_messages = state["extend_data"]["agent_execution_history"][self.record_agent_name]["messages"]
        # Only add new messages that aren't already in the history
        existing_message_ids = {id(msg) for msg in existing_messages}
        new_messages = [msg for msg in current_messages if id(msg) not in existing_message_ids]
        state["extend_data"]["agent_execution_history"][self.record_agent_name]["messages"].extend(new_messages)
        
        return state


class AgentHistoryInjectionMiddleware(AgentMiddleware):
    """
    Middleware that injects agent execution history into messages before agent runs.
    
    This middleware reads metric fault analyst and root cause localizer agent history
    from state and injects them into the message history before the agent executes.
    Uses before_agent hook to inject history into messages.
    
    It converts tool messages to user messages and removes tool_calls from assistant messages
    to prevent the evaluation agent from mistakenly trying to execute those tool calls,
    while still allowing it to see the tool results.
    """
    
    # Define state schema for this middleware
    state_schema = AgentExecutionHistoryState
    
    def __init__(self):
        """Initialize the middleware."""
        super().__init__()
    
    def _filter_history_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Filter history messages to convert tool messages to user messages and remove tool_calls from assistant messages.
        
        Args:
            messages: List of messages from agent execution history
            
        Returns:
            Filtered list of messages suitable for evaluation agent context
        """
        filtered_messages = []
        
        for msg in messages:
            # Determine message type - check both 'type' (langchain) and 'role' (JSON serialized)
            msg_type = None
            if hasattr(msg, 'type'):
                msg_type = msg.type
            elif hasattr(msg, 'role'):
                msg_type = msg.role
            elif isinstance(msg, dict):
                # Handle dict-like messages (from JSON)
                msg_type = msg.get('role') or msg.get('type')
            
            # Convert tool messages to user messages (HumanMessage)
            if msg_type == "tool":
                content = None
                if hasattr(msg, 'content'):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get('content')
                
                if content:
                    filtered_messages.append(AIMessage(content=content))
                # If no content, skip the message
                continue
            
            # For assistant messages with tool_calls, create a new message with only content
            if msg_type in ("ai", "assistant"):
                # Check for tool_calls - check both attribute and dict key
                has_tool_calls = False
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    has_tool_calls = True
                elif isinstance(msg, dict) and msg.get('tool_calls'):
                    has_tool_calls = True
                
                if has_tool_calls:
                    # Create a new AIMessage with only content, no tool_calls
                    content = None
                    if hasattr(msg, 'content'):
                        content = msg.content
                    elif isinstance(msg, dict):
                        content = msg.get('content')
                    
                    if content:
                        filtered_messages.append(HumanMessage(content=content))
                    # If no content, skip the message entirely
                else:
                    # No tool_calls, keep the message as is
                    filtered_messages.append(msg)
            else:
                # For other message types (human, system, etc.), keep as is
                filtered_messages.append(msg)
        
        return filtered_messages
    
    def before_agent(self, state: AgentExecutionHistoryState, config: RunnableConfig) -> AgentExecutionHistoryState:
        """
        Called before agent execution. Inject history into messages.
        Converts tool messages to user messages and removes tool_calls to prevent evaluation agent confusion.
        """
        # Get execution history from state
        execution_history = state.get("extend_data", {}).get("agent_execution_history", {})
        
        # Get histories
        metric_history = execution_history.get("metric_fault_analyst", {})
        rca_history = execution_history.get("root_cause_localizer", {})
        
        # Collect all history messages with filtering
        history_messages = []
        
        # Add metric fault analyst history messages
        if metric_history and "messages" in metric_history:
            metric_messages = metric_history["messages"]
            if isinstance(metric_messages, list):
                history_messages.append(HumanMessage(content="Metric Fault Analyst History:"))
                filtered_metric_messages = self._filter_history_messages(metric_messages)
                history_messages.extend(filtered_metric_messages)
        
        # Add root cause localizer history messages
        if rca_history and "messages" in rca_history:
            rca_messages = rca_history["messages"]
            if isinstance(rca_messages, list):
                history_messages.append(HumanMessage(content="Root Cause Localizer History:"))
                filtered_rca_messages = self._filter_history_messages(rca_messages)
                history_messages.extend(filtered_rca_messages)
        
        # Insert history messages at the beginning
        state["messages"] = history_messages
        
        return state
    
    async def abefore_agent(self, state: AgentExecutionHistoryState, config: RunnableConfig) -> AgentExecutionHistoryState:
        """
        Async version of before_agent. Inject history into messages.
        Converts tool messages to user messages and removes tool_calls to prevent evaluation agent confusion.
        """
        # Get execution history from state
        execution_history = state.get("extend_data", {}).get("agent_execution_history", {})
        
        # Get histories
        metric_history = execution_history.get("metric_fault_analyst", {})
        rca_history = execution_history.get("root_cause_localizer", {})
        
        # Collect all history messages with filtering
        history_messages = []
        
        # Add metric fault analyst history messages
        if metric_history and "messages" in metric_history:
            metric_messages = metric_history["messages"]
            if isinstance(metric_messages, list):
                history_messages.append(HumanMessage(content="Metric Fault Analyst History:"))
                filtered_metric_messages = self._filter_history_messages(metric_messages)
                history_messages.extend(filtered_metric_messages)
        
        # Add root cause localizer history messages
        if rca_history and "messages" in rca_history:
            rca_messages = rca_history["messages"]
            if isinstance(rca_messages, list):
                history_messages.append(HumanMessage(content="Root Cause Localizer History:"))
                filtered_rca_messages = self._filter_history_messages(rca_messages)
                history_messages.extend(filtered_rca_messages)
        
        # Insert history messages at the beginning
        state["messages"] = history_messages
        
        return state

