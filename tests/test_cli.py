import builtins
from unittest.mock import patch, Mock
from click.testing import CliRunner
import os

from ml_analysis_agent.cli import main


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_help_exposes_usage_and_version(self):
        # --help should show usage text and exit 0
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output

        # --version should show version and exit 0
        result = self.runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert 'ML Analysis Agent' in result.output

    @patch('ml_analysis_agent.cli.MLAnalysisAgent')
    @patch('ml_analysis_agent.cli.AWSMLConfig')
    def test_single_query_mode_success(self, mock_config_cls, mock_agent_cls):
        mock_config = Mock()
        mock_config_cls.return_value = mock_config

        mock_agent = Mock()
        mock_agent.load_data.return_value = True
        mock_agent.ask.return_value = "42"
        mock_agent_cls.return_value = mock_agent

        result = self.runner.invoke(main, ['--data', 'some.csv', '--query', 'What is X?'])
        assert result.exit_code == 0
        assert '📊 Result: 42' in result.output
        mock_agent.load_data.assert_called_once_with('some.csv')
        mock_agent.ask.assert_called_once_with('What is X?', verbose=True)
        mock_agent.cleanup.assert_called_once()

    @patch('ml_analysis_agent.cli.MLAnalysisAgent')
    @patch('ml_analysis_agent.cli.AWSMLConfig')
    def test_single_query_missing_data_fails(self, mock_config_cls, mock_agent_cls):
        mock_config = Mock()
        mock_config_cls.return_value = mock_config

        mock_agent = Mock()
        mock_agent_cls.return_value = mock_agent

        result = self.runner.invoke(main, ['--query', 'What is X?'])
        assert result.exit_code != 0
        assert '--data is required' in result.output

    @patch('ml_analysis_agent.cli.MLAnalysisAgent')
    @patch('ml_analysis_agent.cli.AWSMLConfig')
    def test_initialization_failure_exits(self, mock_config_cls, mock_agent_cls):
        mock_config = Mock()
        mock_config_cls.return_value = mock_config

        mock_agent_cls.side_effect = Exception('Initialization failed')

        result = self.runner.invoke(main, [])
        assert result.exit_code != 0
        assert '❌ Error initializing agent' in result.output

    @patch('ml_analysis_agent.cli.MLAnalysisAgent')
    @patch('ml_analysis_agent.cli.AWSMLConfig')
    def test_interactive_quit_flow(self, mock_config_cls, mock_agent_cls):
        mock_config = Mock()
        mock_config_cls.return_value = mock_config

        mock_agent = Mock()
        mock_agent.load_data.return_value = True
        mock_agent_cls.return_value = mock_agent

        inputs = iter(['quit'])
        with patch.object(builtins, 'input', lambda *args: next(inputs)):
            result = self.runner.invoke(main, [])

        assert result.exit_code == 0
        mock_agent.cleanup.assert_called_once()

    def test_missing_aws_token(self):
        # Unset AWS_BEARER_TOKEN_BEDROCK if set
        if 'AWS_BEARER_TOKEN_BEDROCK' in os.environ:
            del os.environ['AWS_BEARER_TOKEN_BEDROCK']

        result = self.runner.invoke(main, [])
        assert result.exit_code != 0
        assert 'AWS_BEARER_TOKEN_BEDROCK environment variable is not set' in result.output