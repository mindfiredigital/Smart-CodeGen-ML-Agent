import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
async def mock_bedrock():
    """Mock AWS Bedrock client for testing."""
    with patch('ml_analysis_agent.agents.base.boto3.client') as mock:
        mock.return_value = AsyncMock()
        mock.return_value.invoke_model = AsyncMock(return_value={
            'body': AsyncMock(read=AsyncMock(return_value=b'{"response": "Test response"}'))
        })
        yield mock

@pytest.fixture
async def mock_llm():
    """Mock LLM responses for testing."""
    with patch('ml_analysis_agent.agents.base.ChatBedrockConverse.agenerate') as mock:
        mock.return_value = "Test response"
        yield mock
