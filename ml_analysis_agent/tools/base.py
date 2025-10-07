"""Base class for all tools in the ML Analysis System."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, *args, **kwargs) -> str:
        """Execute the tool with given parameters."""
        pass

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data. Override in subclasses if needed."""
        return True

    def handle_error(self, error: Exception) -> str:
        """Handle errors in a consistent way."""
        import traceback

        return f'❌ Error in {self.name}: {str(error)}\n{traceback.format_exc()}'

    def format_success(self, message: str) -> str:
        """Format success messages consistently."""
        return f'✅ {message}'

    def format_failure(self, message: str) -> str:
        """Format failure messages consistently."""
        return f'❌ {message}'
