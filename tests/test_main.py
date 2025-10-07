import pytest
from unittest.mock import Mock, patch
from ml_analysis_agent.main import main


def test_main_loads_data_and_quit(monkeypatch):
    """Verify main loads data and calls cleanup when user quits."""
    mock_config = Mock()
    mock_agent = Mock()
    mock_agent.load_data.return_value = True

    # sequence: provide data file, then quit at the question prompt
    inputs = iter(["some.csv", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt='': next(inputs))

    with patch("ml_analysis_agent.main.AWSMLConfig", return_value=mock_config) as mock_conf, \
         patch("ml_analysis_agent.main.MLAnalysisAgent", return_value=mock_agent) as mock_agent_cls:
        main()

    mock_conf.assert_called_once()
    mock_agent_cls.assert_called_once_with(mock_config)
    mock_agent.load_data.assert_called_once_with("some.csv")
    mock_agent.cleanup.assert_called_once()


def test_main_invalid_then_valid(monkeypatch):
    """If first load_data fails, main should prompt again and succeed on second attempt."""
    mock_config = Mock()
    mock_agent = Mock()
    # first call returns False (invalid file), second returns True
    mock_agent.load_data.side_effect = [False, True]

    inputs = iter(["bad.csv", "good.csv", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt='': next(inputs))

    with patch("ml_analysis_agent.main.AWSMLConfig", return_value=mock_config), \
         patch("ml_analysis_agent.main.MLAnalysisAgent", return_value=mock_agent):
        main()

    assert mock_agent.load_data.call_count == 2
    mock_agent.cleanup.assert_called_once()


def test_main_change_data_flow(monkeypatch):
    """Simulate 'change-data' command: initial load, change-data, load new file, then quit."""
    mock_config = Mock()
    mock_agent = Mock()
    # initial load True, new data load True
    mock_agent.load_data.side_effect = [True, True]

    inputs = iter(["initial.csv", "change-data", "new.csv", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt='': next(inputs))

    with patch("ml_analysis_agent.main.AWSMLConfig", return_value=mock_config), \
         patch("ml_analysis_agent.main.MLAnalysisAgent", return_value=mock_agent):
        main()

    # two successful load_data calls: initial and after change-data
    assert mock_agent.load_data.call_count == 2
    mock_agent.cleanup.assert_called_once()