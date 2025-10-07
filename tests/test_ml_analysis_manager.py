import pytest
from unittest.mock import patch, Mock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from ml_analysis_agent.ml_analysis_manager import MLAnalysisManager
from ml_analysis_agent.config.ml_config import MLConfig
from ml_analysis_agent.config.file_config import FileConfig


@pytest.fixture
def mock_llm():
    """Fixture to mock LLM model."""
    mock = Mock()
    return mock

@pytest.fixture
def mock_agents():
    """Fixture to mock both CodeGeneratorAgent and CodeExecutorAgent."""
    with patch('ml_analysis_agent.ml_analysis_manager.CodeGeneratorAgent') as mock_gen, \
         patch('ml_analysis_agent.ml_analysis_manager.CodeExecutorAgent') as mock_exec:
        mock_gen.return_value = Mock(name="code_generator")
        mock_exec.return_value = Mock(name="code_executor")
        yield mock_gen, mock_exec

@pytest.fixture
def mock_supervisor():
    """Fixture to mock SupervisorManager."""
    with patch('ml_analysis_agent.ml_analysis_manager.SupervisorManager') as mock:
        mock.return_value = Mock()
        yield mock

class TestMLAnalysisManager:
    """Test the MLAnalysisManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ml_config = MLConfig()
        self.file_config = FileConfig()
        
        # Mock get_llm_model
        self.ml_config.get_llm_model = Mock()
        self.ml_config.get_llm_model.return_value = Mock()
    
    def test_manager_initialization(self, mock_agents, mock_supervisor):
        """Test MLAnalysisManager initialization."""
        # Unpack the mocked agents
        mock_generator, mock_executor = mock_agents
        
        manager = MLAnalysisManager(ml_config=self.ml_config)
        assert manager.ml_config == self.ml_config
        assert isinstance(manager.file_config, FileConfig)
        
        # Verify agents were created
        mock_generator.assert_called_once()
        mock_executor.assert_called_once()
        mock_supervisor.assert_called_once()
    
    def test_pretty_print_message(self, mock_agents, mock_supervisor):
        """Test pretty printing a single message."""
        manager = MLAnalysisManager(ml_config=self.ml_config)
        mock_message = HumanMessage(content="Test message")
        
        with patch('builtins.print') as mock_print, \
             patch.object(mock_message, 'pretty_repr', return_value="Test message"):
            manager.pretty_print_message(mock_message)
            mock_print.assert_called_with("Test message")
    
    def test_pretty_print_messages(self, mock_agents, mock_supervisor):
        """Test pretty printing multiple messages."""
        manager = MLAnalysisManager(ml_config=self.ml_config)
        mock_update = {
            "node1": {
                "messages": [{"role": "user", "content": "Test content"}]
            }
        }
        
        with patch('builtins.print') as mock_print, \
             patch('langchain_core.messages.convert_to_messages') as mock_convert:
            mock_convert.return_value = [HumanMessage(content="Test content")]
            manager.pretty_print_messages(mock_update)
            mock_print.assert_called()
    
    def test_run_analysis(self, mock_agents, mock_supervisor):
        """Test running analysis."""
        manager = MLAnalysisManager(ml_config=self.ml_config)
        mock_response = AIMessage(content="Test response")
        
        mock_supervisor_instance = mock_supervisor.return_value
        mock_supervisor_instance.stream.return_value = iter([{
            "node1": {
                "messages": [{"role": "assistant", "content": "Test response"}]
            }
        }])
        
        with patch('langchain_core.messages.convert_to_messages') as mock_convert:
            mock_convert.return_value = [mock_response]
            result = manager.run_analysis("Test query")
            assert result == "Test response"
            mock_supervisor_instance.stream.assert_called_once()
    
    def test_run_analysis_with_file(self, mock_agents, mock_supervisor):
        """Test running analysis with current data file."""
        manager = MLAnalysisManager(ml_config=self.ml_config)
        manager.file_config.set_current_data_file("test.csv")
        
        mock_response = AIMessage(content="Test response")
        mock_supervisor_instance = mock_supervisor.return_value
        mock_supervisor_instance.stream.return_value = iter([{
            "node1": {
                "messages": [{"role": "assistant", "content": "Test response"}]
            }
        }])
        
        with patch('langchain_core.messages.convert_to_messages') as mock_convert:
            mock_convert.return_value = [mock_response]
            result = manager.run_analysis("Analyze this")
            assert "Test response" in result
            call_args = mock_supervisor_instance.stream.call_args[0][0]
            assert "test.csv" in call_args["messages"][0]["content"]
    
    def test_load_data_file(self, mock_agents, mock_supervisor):
        """Test loading data file."""
        manager = MLAnalysisManager(ml_config=self.ml_config)
        mock_file_manager = Mock()
        mock_file_manager.validate_and_copy_data_file.return_value = (True, "Success")
        manager.file_manager = mock_file_manager
        
        success, message = manager.load_data_file("test.csv")
        
        assert success is True
        assert message == "Success"
        mock_file_manager.validate_and_copy_data_file.assert_called_once_with("test.csv")
