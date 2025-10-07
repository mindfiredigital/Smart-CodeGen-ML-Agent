"""Base agent class for the ML Analysis System."""

from abc import ABC, abstractmethod
from typing import List, Any
from langgraph.prebuilt import create_react_agent


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str, model, tools: List[Any], prompt: str):
        self.name = name
        self.model = model
        self.tools = tools
        self.prompt = prompt
        self._agent = None

    def create_agent(self):
        """Create the actual agent using langgraph."""
        if self._agent is None:
            self._agent = create_react_agent(
                model=self.model,
                tools=self.tools,
                prompt=self.prompt,
                name=self.name,
            )
        return self._agent

    @abstractmethod
    def get_prompt(self) -> str:
        """Get the agent-specific prompt."""
        pass

    @abstractmethod
    def get_tools(self) -> List[Any]:
        """Get the agent-specific tools."""
        pass
