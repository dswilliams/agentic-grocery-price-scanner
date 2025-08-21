"""
Base agent class for the grocery price scanner system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..utils import get_logger


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str):
        """Initialize the base agent."""
        self.name = name
        self.settings = get_settings()
        self.logger = get_logger(f"agent.{name}")
        self.state = {}
    
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main functionality."""
        pass
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str, exc_info: Optional[Exception] = None) -> None:
        """Log an error message."""
        self.logger.error(f"[{self.name}] {message}", exc_info=exc_info)
    
    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(f"[{self.name}] {message}")
    
    def update_state(self, key: str, value: Any) -> None:
        """Update agent state."""
        self.state[key] = value
        self.log_debug(f"Updated state: {key} = {value}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from agent state."""
        return self.state.get(key, default)