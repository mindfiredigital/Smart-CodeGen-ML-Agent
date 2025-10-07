"""
ML Analysis Agent - A multi-agent system for ML analysis using AWS Bedrock and Claude.
This package provides a simple interface for generating and executing ML code
based on natural language queries.
"""

__version__ = '0.1.0'
__author__ = 'Dipika Dhara'
__license__ = 'MIT'

from pathlib import Path
from .ml_analysis_manager import MLAnalysisManager
from .config.file_config import FileConfig
from .config.ml_config import MLConfig
from .config.prompt_manager import PromptManager, get_prompt_manager


# Create a user-friendly wrapper class
class MLAnalysisAgent:
    """
    Main interface for the ML Analysis Agent.

    This class provides a simple library for users to interact with the multi-agent system.

    Args:
        aws_token (str, optional): AWS bearer token for Bedrock. If not provided,
            will look for AWS_BEARER_TOKEN_BEDROCK environment variable.
        aws_region (str, optional): AWS region. Defaults to 'us-west-2'.
        output_dir (str, optional): Directory for generated code. Defaults to 'temp'.
        data_dir (str, optional): Directory for data files. Defaults to 'data'.
    """

    def __init__(self, ml_config: MLConfig = None, output_dir: str = None, data_dir: str = None):
        """Initialize the ML Analysis Agent."""
        import os

        self.ml_config = ml_config
        self._manager = MLAnalysisManager(self.ml_config)

        # Override directories if provided
        if output_dir:
            self._manager.file_config.OUTPUT_DIR = Path(output_dir)
        else:
            self._manager.file_config.OUTPUT_DIR = Path('temp')
        self._manager.file_config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if data_dir:
            self._manager.file_config.CSV_DATA_DIR = Path(data_dir)
        else:
            self._manager.file_config.CSV_DATA_DIR = Path('data')
        self._manager.file_config.CSV_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Flag to track if data is loaded
        self._data_loaded = False

    def load_data(self, file_path: str) -> bool:
        """
        Load a data file for analysis.

        Args:
            file_path (str): Path to the data file (CSV, Excel, JSON, or Parquet)

        Returns:
            bool: True if successful, False otherwise

        """
        success, message = self._manager.load_data_file(file_path)
        if success:
            self._data_loaded = True
        print(message)
        return success

    def ask(self, question: str, verbose: bool = True) -> str:
        """
        Ask a question about the loaded data.

        Args:
            question (str): Natural language question about the data
            verbose (bool): Whether to print intermediate steps. Defaults to True.

        Returns:
            str: The answer to the question
        """
        if not self._data_loaded:
            raise ValueError('No data loaded. Please load data using load_data() first.')

        if not verbose:
            # Suppress intermediate output
            import sys
            import io

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        try:
            # Pass verbose flag to run_analysis
            result = self._manager.run_analysis(question, verbose=verbose)

            if not verbose:
                sys.stdout = old_stdout

            return result
        except Exception as e:
            if not verbose:
                sys.stdout = old_stdout
            raise e

    def ask_stream(self, question: str):
        """
        Ask a question and stream the response.

        Args:
            question (str): Natural language question about the data

        Yields:
            dict: Chunks of the response as they are generated
        """
        current_file = self._manager.file_config.get_current_data_file()
        if current_file and not any(
            ext in question.lower() for ext in ['.csv', '.xlsx', '.json', '.parquet']
        ):
            question = f'{question} using dataset at {current_file}'

        for chunk in self._manager.supervisor_manager.stream(
            {'messages': [{'role': 'user', 'content': question}]}
        ):
            yield chunk

    def _extract_final_answer(self, result) -> str:
        """Extract the final answer from the result."""
        if not result:
            return 'No result available'

        # Try to extract the last message content
        try:
            # Result is typically a dict with messages
            for node_name, node_data in result.items():
                if 'messages' in node_data:
                    messages = node_data['messages']
                    if messages:
                        last_message = messages[-1]
                        if hasattr(last_message, 'content'):
                            return last_message.content
                        elif isinstance(last_message, dict) and 'content' in last_message:
                            return last_message['content']
        except Exception:
            pass

        return str(result)

    def get_data_info(self) -> dict:
        """
        Get information about the currently loaded data.

        Returns:
            dict: Information about the data file
        """
        from .tools.csv_analyzer import CSVAnalyzer

        current_file = self._manager.file_config.get_current_data_file()
        if not current_file:
            return {'error': 'No data file loaded'}

        analyzer = CSVAnalyzer(self._manager.file_config)
        result = analyzer.execute(current_file)

        # Parse the JSON result
        import json

        try:
            # Extract JSON from the result string
            json_start = result.index('{')
            json_str = result[json_start:]
            return json.loads(json_str)
        except Exception:
            return {'error': 'Could not parse data info'}

    def cleanup(self):
        """
        Clean up temporary files and directories.

        Example:
            >>> agent.cleanup()
        """
        if self._data_loaded:
            self._manager.file_manager.cleanup_data_folder()
            self._data_loaded = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()
        return False


# Convenience function for quick usage
def create_agent(**kwargs) -> MLAnalysisAgent:
    """
    Create an ML Analysis Agent with the given configuration.

    Args:
        **kwargs: Arguments to pass to MLAnalysisAgent

    Returns:
        MLAnalysisAgent: Configured agent instance

    Example:
        >>> agent = create_agent(aws_token="token123")
    """
    return MLAnalysisAgent(**kwargs)


# Export public API
__all__ = [
    'MLAnalysisAgent',
    'create_agent',
    'MLAnalysisManager',
    'FileConfig',
    'MLConfig',
    'PromptManager',
    'get_prompt_manager',
    '__version__',
]
