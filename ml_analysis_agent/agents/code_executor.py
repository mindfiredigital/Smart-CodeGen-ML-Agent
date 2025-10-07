"""Code executor agent for running generated ML code."""

from typing import List, Any
from .base import BaseAgent
from ..tools.code_executor import code_executor_tool
from ..config.prompt_manager import get_prompt_manager


class CodeExecutorAgent(BaseAgent):
    """Agent responsible for executing generated ML code."""

    def __init__(self, model):
        self.prompt_manager = get_prompt_manager()
        super().__init__(
            name='code_executor_agent',
            model=model,
            tools=self.get_tools(),
            prompt=self.get_prompt(),
        )

    def get_tools(self) -> List[Any]:
        """Get tools specific to code execution."""
        return [code_executor_tool]

    def get_prompt(self) -> str:
        """Get the prompt for code execution."""
        # Use PromptManager to get the prompt
        return self.prompt_manager.get_prompt('code_executor')
