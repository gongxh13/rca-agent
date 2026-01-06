"""
Base RCA Tool

Provides the abstract base class for all RCA analysis tools. This class defines
the common interface and shared functionality for log, trace, and metric analysis tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import inspect

from langchain.tools import tool
from langchain_core.tools import StructuredTool


class BaseRCATool(ABC):
    """
    Abstract base class for RCA analysis tools.
    
    All RCA tools (log, trace, metric) should inherit from this class and implement
    the get_tools() method to return a list of LangChain tools.
    
    The design philosophy is to provide high-level semantic operations rather than
    raw data access, which helps minimize context size and makes the tools more
    useful for LLM-based agents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RCA tool.
        
        Args:
            config: Optional configuration dictionary for the tool.
                   Can include data source paths, connection settings, etc.
        """
        self.config = config or {}
        self._initialized = False
        
    def initialize(self) -> None:
        """
        Initialize the tool and establish connections to data sources.
        
        This method should be called before using any analysis methods.
        Subclasses can override this to perform specific initialization tasks.
        """
        self._initialized = True
        
    def is_initialized(self) -> bool:
        """Check if the tool has been initialized."""
        return self._initialized
    
    @abstractmethod
    def get_tools(self) -> List[Any]:
        """
        Get a list of LangChain tools provided by this RCA tool.
        
        Returns:
            List of LangChain tool objects
        """
        pass
    
    def validate_time_range(
        self, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> tuple[datetime, datetime]:
        """
        Validate and normalize time range parameters.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            
        Returns:
            Tuple of (start_time, end_time) with defaults applied if needed
            
        Raises:
            ValueError: If time range is invalid
        """
        if start_time and end_time and start_time > end_time:
            raise ValueError("start_time must be before end_time")
            
        # Apply defaults if needed
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            # Default to 1 hour before end_time
            from datetime import timedelta
            start_time = end_time - timedelta(hours=1)
            
        return start_time, end_time
    
    def _build_structured_tool(self, func: Any, name: Optional[str] = None, description: Optional[str] = None) -> StructuredTool:
        """
        Wrap a bound method into a StructuredTool with propagated description.
        
        Description propagation order:
        1) description argument (if provided)
        2) func.description (if func is already a Tool)
        3) func.__doc__ from subclass implementation
        4) Base class method description (if Base method is a Tool)
        5) Base class method docstring
        """
        method_name = name or getattr(func, "name", None) or func.__name__
        
        docstring = description
        
        # 1. Check func itself (child implementation or inherited tool)
        if not docstring:
            if hasattr(func, "description") and func.description:
                docstring = func.description
            elif hasattr(func, "__doc__") and func.__doc__:
                docstring = func.__doc__
        
        # 2. Check parent if still no docstring
        if not docstring:
            try:
                # Use super() to find parent implementation
                parent_method = getattr(super(type(self), self), method_name, None)
                if parent_method:
                    if hasattr(parent_method, "description") and parent_method.description:
                        docstring = parent_method.description
                    elif hasattr(parent_method, "__doc__") and parent_method.__doc__:
                        docstring = parent_method.__doc__
            except Exception:
                pass

        return StructuredTool.from_function(
            func=func,
            name=method_name,
            description=docstring.strip() if docstring else method_name
        )
    
    def cleanup(self) -> None:
        """
        Clean up resources and close connections.
        
        Subclasses should override this to perform cleanup tasks.
        """
        self._initialized = False
