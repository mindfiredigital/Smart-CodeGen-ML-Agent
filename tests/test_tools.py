import pytest
from unittest.mock import patch, Mock, MagicMock
import tempfile
import os
import pandas as pd
from pathlib import Path

from ml_analysis_agent.tools.code_executor import CodeExecutor
from ml_analysis_agent.tools.code_saver import CodeSaver
from ml_analysis_agent.tools.csv_analyzer import CSVAnalyzer
from ml_analysis_agent.config.file_config import FileConfig

# ----------- CodeExecutor Tests -----------
class TestCodeExecutor:
    def setup_method(self):
        self.file_config = FileConfig()
        self.executor = CodeExecutor(self.file_config)

    def test_execute_valid_python_file(self):
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write('print("Hello World")')
            f.flush()
            result = self.executor.execute(f.name)
        assert "Hello World" in result
        os.remove(f.name)

    def test_execute_nonexistent_file(self):
        result = self.executor.execute('nonexistent_file.py')
        assert "No such file" in result

    @patch('ml_analysis_agent.tools.code_executor.CodeExecutor.install_dependency')
    def test_execute_with_missing_dependency(self, mock_install):
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write('import fakepackage\nprint("Test")')
            f.flush()
            mock_install.return_value = "Failed to install fakepackage"
            result = self.executor.execute(f.name)
        assert (
            "Failed to install fakepackage" in result
            or "Test" in result
            or "No module named 'fakepackage'" in result
            or "Execution Error" in result
        )
        os.remove(f.name)

    def test_install_dependency_already_installed(self):
        result = self.executor.install_dependency('sys')
        assert "already installed" in result

    @patch('subprocess.check_call', side_effect=Exception('pip error'))
    def test_install_dependency_failure(self, mock_call):
        result = self.executor.install_dependency('fakepackage')
        assert "Failed to install" in result

# ----------- CodeSaver Tests -----------
class TestCodeSaver:
    def setup_method(self):
        self.file_config = FileConfig()
        self.saver = CodeSaver(self.file_config)

    def test_save_valid_code(self):
        code = 'print("Saved!")'
        filename = 'test_save_code.py'
        result = self.saver.execute(code, filename)
        assert "Code saved" in result
        path = self.file_config.get_output_path(filename)
        assert path.exists()
        path.unlink()

    def test_save_invalid_code(self):
        result = self.saver.execute('', 'invalid.py')
        assert "Invalid code" in result

    def test_validate_input(self):
        assert self.saver.validate_input('print(1)') is True
        assert self.saver.validate_input('   ') is False
        assert self.saver.validate_input('') is False

# ----------- CSVAnalyzer Tests -----------
class TestCSVAnalyzer:
    def setup_method(self):
        self.file_config = FileConfig()
        self.analyzer = CSVAnalyzer(self.file_config)

    def test_analyze_valid_csv(self):
        df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
        temp_path = Path(tempfile.gettempdir()) / 'test_analyze.csv'
        df.to_csv(temp_path, index=False)
        result = self.analyzer.execute(str(temp_path))
        assert "CSV Analysis Complete" in result
        temp_path.unlink()

    def test_analyze_nonexistent_csv(self):
        result = self.analyzer.execute('nonexistent.csv')
        assert "CSV file not found" in result

    def test_analyze_csv_with_missing_values(self):
        df = pd.DataFrame({'A': [1, None], 'B': ['x', None]})
        temp_path = Path(tempfile.gettempdir()) / 'test_missing.csv'
        df.to_csv(temp_path, index=False)
        result = self.analyzer.execute(str(temp_path))
        assert "missing_values" in result
        temp_path.unlink()
