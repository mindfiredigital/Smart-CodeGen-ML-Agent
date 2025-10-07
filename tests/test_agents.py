import pytest
from unittest.mock import patch, Mock
from ml_analysis_agent.agents.code_executor import CodeExecutorAgent
from ml_analysis_agent.agents.code_generator import CodeGeneratorAgent
from ml_analysis_agent.config.file_config import FileConfig

class TestCodeExecutorAgent:
    @patch('ml_analysis_agent.agents.code_executor.get_prompt_manager')
    def test_initialization_and_tools(self, mock_prompt_manager):
        mock_model = Mock()
        mock_prompt_manager.return_value.get_prompt.return_value = "EXECUTOR_PROMPT"
        agent = CodeExecutorAgent(mock_model)
        assert agent.name == "code_executor_agent"
        assert agent.model is mock_model
        assert agent.get_tools()[0].name == "code_executor_tool"
        assert agent.get_prompt() == "EXECUTOR_PROMPT"

class TestCodeGeneratorAgent:
    @patch('ml_analysis_agent.agents.code_generator.get_prompt_manager')
    def test_initialization_and_tools(self, mock_prompt_manager):
        mock_model = Mock()
        mock_file_config = Mock(spec=FileConfig)
        mock_file_config.get_current_data_file.return_value = "data.csv"
        mock_prompt_manager.return_value.get_prompt.return_value = "GENERATOR_PROMPT"
        agent = CodeGeneratorAgent(mock_model, mock_file_config)
        assert agent.name == "code_generator_agent"
        assert agent.model is mock_model
        tools = agent.get_tools()
        assert any(t.name == "csv_analyzer" for t in tools)
        assert any(t.name == "save_code_tool" for t in tools)
        assert agent.get_prompt() == "GENERATOR_PROMPT"

    @patch('ml_analysis_agent.agents.code_generator.get_prompt_manager')
    def test_prompt_with_no_data_file(self, mock_prompt_manager):
        mock_model = Mock()
        mock_file_config = Mock(spec=FileConfig)
        mock_file_config.get_current_data_file.return_value = None
        mock_prompt_manager.return_value.get_prompt.return_value = "GENERATOR_PROMPT"
        agent = CodeGeneratorAgent(mock_model, mock_file_config)
        prompt = agent.get_prompt()
        assert prompt == "GENERATOR_PROMPT"
