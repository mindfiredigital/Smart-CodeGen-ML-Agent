import pytest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path

from ml_analysis_agent import MLAnalysisAgent
from ml_analysis_agent.config.ml_config import MLConfig

@pytest.fixture
def mock_llm():
    with patch('ml_analysis_agent.config.ml_config.MLConfig.get_llm_model') as mock:
        yield mock

@pytest.fixture
def mock_manager():
    with patch('ml_analysis_agent.MLAnalysisManager') as mock:
        yield mock

class TestMLAnalysisAgent:
    def setup_method(self):
        """Set up test fixtures."""
        self.ml_config = MLConfig()
        self.test_output_dir = "test_output"
        self.test_data_dir = "test_data"
    
    def test_agent_initialization(self, mock_llm, mock_manager):
        """Test MLAnalysisAgent can be initialized with config."""
        agent = MLAnalysisAgent(
            ml_config=self.ml_config,
            output_dir=self.test_output_dir,
            data_dir=self.test_data_dir
        )
        assert agent is not None
        assert agent.ml_config == self.ml_config
        mock_manager.assert_called_once()
        manager_instance = mock_manager.return_value
        assert str(manager_instance.file_config.OUTPUT_DIR) == self.test_output_dir
    
    def test_agent_with_default_paths(self, mock_llm, mock_manager):
        """Test agent initialization with default paths."""
        agent = MLAnalysisAgent(ml_config=self.ml_config)
        assert agent is not None
        mock_manager.assert_called_once()
        manager_instance = mock_manager.return_value
        assert str(manager_instance.file_config.OUTPUT_DIR) == "temp"
        assert str(manager_instance.file_config.CSV_DATA_DIR) == "data"
    
    def test_data_loading(self, mock_llm, mock_manager):
        """Test data loading functionality."""
        mock_manager_instance = MagicMock()
        mock_manager_instance.load_data_file.return_value = (True, "Success message")
        mock_manager.return_value = mock_manager_instance
        
        agent = MLAnalysisAgent(ml_config=self.ml_config)
        test_file = "test/test_data/Housing.csv"
        
        success = agent.load_data(test_file)
        assert success is True
        assert agent._data_loaded is True
        mock_manager_instance.load_data_file.assert_called_once_with(test_file)
    
    def test_cleanup(self, mock_llm, mock_manager):
        """Test cleanup functionality."""
        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance
        
        agent = MLAnalysisAgent(ml_config=self.ml_config)
        agent._data_loaded = True
        agent.cleanup()
        
        assert agent._data_loaded is False
        mock_manager_instance.file_manager.cleanup_data_folder.assert_called_once()
    
    def test_ask_question(self, mock_llm, mock_manager):
        """Test asking questions."""
        mock_manager_instance = MagicMock()
        mock_manager_instance.run_analysis.return_value = "Test response"
        mock_manager.return_value = mock_manager_instance
        
        agent = MLAnalysisAgent(ml_config=self.ml_config)
        agent._data_loaded = True
        
        result = agent.ask("Test question")
        assert result == "Test response"
        mock_manager_instance.run_analysis.assert_called_once_with("Test question", verbose=True)
    
    def test_ask_without_data(self, mock_llm, mock_manager):
        """Test asking question without loading data."""
        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance
        
        agent = MLAnalysisAgent(ml_config=self.ml_config)
        
        with pytest.raises(ValueError) as exc_info:
            agent.ask("Test question")
        assert "No data loaded" in str(exc_info.value)
    
    @patch('ml_analysis_agent.MLAnalysisManager')
    def test_ask_stream(self, mock_manager):
        """Test stream response functionality."""
        mock_manager_instance = MagicMock()
        mock_manager_instance.supervisor_manager.stream.return_value = iter(["chunk1", "chunk2"])
        mock_manager_instance.file_config.get_current_data_file.return_value = "test.csv"
        mock_manager.return_value = mock_manager_instance
        
        agent = MLAnalysisAgent(ml_config=self.ml_config)
        chunks = list(agent.ask_stream("Test question"))
        
        assert chunks == ["chunk1", "chunk2"]
        mock_manager_instance.supervisor_manager.stream.assert_called_once()
    
    @patch('ml_analysis_agent.MLAnalysisManager')
    def test_get_data_info(self, mock_manager):
        """Test getting data info."""
        mock_manager_instance = MagicMock()
        mock_manager_instance.file_config.get_current_data_file.return_value = "test.csv"
        mock_manager.return_value = mock_manager_instance
        
        agent = MLAnalysisAgent(ml_config=self.ml_config)
        info = agent.get_data_info()
        
        assert isinstance(info, dict)
        # Should return error dict if no data file is loaded
        mock_manager_instance.file_config.get_current_data_file.return_value = None
        info = agent.get_data_info()
        assert "error" in info
