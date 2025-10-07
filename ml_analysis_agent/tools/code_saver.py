"""Code saver tool for saving generated Python code."""

from langchain.tools import tool
from .base import BaseTool


class CodeSaver(BaseTool):
    """Tool for saving generated Python code to files."""

    def __init__(self, file_config):
        super().__init__(name='save_code_tool', description='Save generated Python code')
        self.file_config = file_config

    def execute(self, code: str, filename: str = 'ml_analysis.py') -> str:
        """Save generated Python code to a file."""
        try:
            if not self.validate_input(code):
                return self.format_failure('Invalid code provided')

            file_path = self.file_config.get_output_path(filename)
            file_path.write_text(code)
            return self.format_success(f'Code saved to {file_path}')

        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, code: str) -> bool:
        """Validate that code is a non-empty string."""
        return isinstance(code, str) and len(code.strip()) > 0


@tool('save_code_tool')
def save_code_tool(code: str) -> str:
    """Save generated Python code."""
    from ..config.file_config import FileConfig

    file_config = FileConfig()
    saver = CodeSaver(file_config)
    return saver.execute(code)
