"""Code generator agent for creating ML analysis code."""

from typing import List, Any
from .base import BaseAgent
from ..tools.csv_analyzer import csv_analyzer_tool
from ..tools.code_saver import save_code_tool
from ..config.prompt_manager import get_prompt_manager


class CodeGeneratorAgent(BaseAgent):
    """Agent responsible for generating ML analysis code."""

    def __init__(self, model, file_config):
        self.file_config = file_config
        self.prompt_manager = get_prompt_manager()
        super().__init__(
            name='code_generator_agent',
            model=model,
            tools=self.get_tools(),
            prompt=self.get_prompt(),
        )

    def get_tools(self) -> List[Any]:
        """Get tools specific to code generation."""
        return [csv_analyzer_tool, save_code_tool]

    def get_prompt(self) -> str:
        """Get the prompt for code generation."""
        current_data_file = self.file_config.get_current_data_file() or '{CURRENT_DATA_FILE}'

        # Use PromptManager to get the prompt
        return self.prompt_manager.get_prompt('code_generator', current_data_file=current_data_file)
