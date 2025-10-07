import pytest
from unittest.mock import patch, Mock, mock_open, MagicMock
import os
from pathlib import Path

from ml_analysis_agent.file_manager import FileManager
from ml_analysis_agent.config.file_config import FileConfig


class TestFileManager:
    """Test the FileManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.file_config = FileConfig()
        self.file_manager = FileManager(file_config=self.file_config)
    
    def test_validate_file_extension(self):
        """Test validating file extension."""
        test_file = Path("test.csv")
        self.file_config.VALID_EXTENSIONS = ['.csv']
        assert self.file_manager.validate_file_extension(test_file) is True
        
        test_file = Path("test.txt")
        assert self.file_manager.validate_file_extension(test_file) is False
    
    @patch('pathlib.Path.exists')
    @patch('shutil.copy2')
    def test_validate_and_copy_data_file(self, mock_copy, mock_exists):
        """Test validating and copying data file."""
        mock_exists.return_value = True
        test_file = "test.csv"
        self.file_config.VALID_EXTENSIONS = ['.csv']
        
        success, message = self.file_manager.validate_and_copy_data_file(test_file)
        assert success is True
        assert "successfully" in message
        mock_copy.assert_called_once()
    
    @patch('pathlib.Path.exists')
    def test_validate_and_copy_data_file_not_exists(self, mock_exists):
        """Test validating non-existent file."""
        mock_exists.return_value = False
        test_file = "nonexistent.csv"
        
        success, message = self.file_manager.validate_and_copy_data_file(test_file)
        assert success is False
        assert "not found" in message.lower()
    
    @patch('pathlib.Path.exists')
    def test_validate_and_copy_data_file_invalid_extension(self, mock_exists):
        """Test validating file with invalid extension."""
        mock_exists.return_value = True
        test_file = "test.txt"
        self.file_config.VALID_EXTENSIONS = ['.csv']
        
        success, message = self.file_manager.validate_and_copy_data_file(test_file)
        assert success is False
        assert "unsupported" in message.lower()
    
    @patch('shutil.rmtree')
    @patch('pathlib.Path.exists')
    def test_cleanup_data_folder(self, mock_exists, mock_rmtree):
        """Test cleanup of data folder."""
        mock_exists.return_value = True
        
        self.file_manager.cleanup_data_folder()
        mock_rmtree.assert_called_once_with(self.file_config.CSV_DATA_DIR)
    
    @patch('atexit.register')
    def test_register_cleanup(self, mock_register):
        """Test cleanup registration."""
        # Create a new FileManager to trigger registration
        file_manager = FileManager(file_config=self.file_config)
        
        # Check if cleanup_data_folder was registered with atexit
        mock_register.assert_called_once_with(file_manager.cleanup_data_folder)
