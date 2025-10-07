import pytest
from unittest.mock import patch, Mock, MagicMock
import os
import boto3
from botocore.exceptions import ClientError

from ml_analysis_agent.config.ml_config import MLConfig, AWSMLConfig


class TestMLConfig:
    """Test the base MLConfig class."""
    
    def test_base_config_get_llm_model_not_implemented(self):
        """Test that base MLConfig raises NotImplementedError for get_llm_model."""
        config = MLConfig()
        
        with pytest.raises(NotImplementedError):
            config.get_llm_model()
    
    def test_base_config_get_client(self):
        """Test that base MLConfig get_client method returns None."""
        config = MLConfig()
        result = config.get_client()
        assert result is None


class TestAWSMLConfig:
    """Test the AWSMLConfig class."""
    
    def test_config_initialization_with_params(self):
        """Test AWSMLConfig initialization with direct parameters."""
        config = AWSMLConfig(
            aws_token="test_token",
            aws_region="us-west-2",
            model_name="test.model"
        )
        
        assert config.AWS_BEARER_TOKEN_BEDROCK == "test_token"
        assert config.AWS_DEFAULT_REGION == "us-west-2"
        assert config.MODEL_NAME == "test.model"
    
    def test_config_initialization_from_env(self):
        """Test AWSMLConfig initialization from environment variables."""
        with patch.dict(os.environ, {
            'AWS_BEARER_TOKEN_BEDROCK': 'env_token',
            'AWS_DEFAULT_REGION': 'us-east-1',
            'MODEL_NAME': 'env.model'
        }):
            config = AWSMLConfig()
            
            assert config.AWS_BEARER_TOKEN_BEDROCK == "env_token"
            assert config.AWS_DEFAULT_REGION == "us-east-1"
            assert config.MODEL_NAME == "env.model"
    
    def test_config_default_values(self):
        """Test AWSMLConfig default values when not provided."""
        with patch.dict(os.environ, {
            'AWS_BEARER_TOKEN_BEDROCK': 'test_token'
        }, clear=True):
            config = AWSMLConfig()
            
            assert config.AWS_BEARER_TOKEN_BEDROCK == "test_token"
            assert config.AWS_DEFAULT_REGION == "us-west-2"  # default
            assert config.MODEL_NAME == "us.anthropic.claude-sonnet-4-20250514-v1:0"  # default
    
    def test_missing_aws_token_raises_error(self):
        """Test that missing AWS token raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="AWS_BEARER_TOKEN_BEDROCK not found"):
                AWSMLConfig()
    
    def test_parameter_override_env(self):
        """Test that direct parameters override environment variables."""
        with patch.dict(os.environ, {
            'AWS_BEARER_TOKEN_BEDROCK': 'env_token',
            'AWS_DEFAULT_REGION': 'us-east-1',
            'MODEL_NAME': 'env.model'
        }):
            config = AWSMLConfig(
                aws_token="param_token",
                aws_region="us-west-1",
                model_name="param.model"
            )
            
            assert config.AWS_BEARER_TOKEN_BEDROCK == "param_token"
            assert config.AWS_DEFAULT_REGION == "us-west-1"
            assert config.MODEL_NAME == "param.model"
    
    @patch('ml_analysis_agent.config.ml_config.boto3.client')
    def test_get_client_singleton(self, mock_boto_client):
        """Test that get_client returns the same instance (singleton pattern)."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        config = AWSMLConfig(aws_token="test_token")
        
        # First call
        client1 = config.get_client()
        # Second call
        client2 = config.get_client()
        
        assert client1 is client2
        assert client1 is mock_client
        # boto3.client should only be called once due to singleton pattern
        mock_boto_client.assert_called_once_with(
            "bedrock-runtime",
            region_name="us-west-2",
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token="test_token",
        )
    
    @patch('ml_analysis_agent.config.ml_config.ChatBedrockConverse')
    @patch('ml_analysis_agent.config.ml_config.boto3.client')
    def test_get_llm_model_singleton(self, mock_boto_client, mock_chat_bedrock):
        """Test that get_llm_model returns the same instance (singleton pattern)."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_llm = Mock()
        mock_chat_bedrock.return_value = mock_llm
        
        config = AWSMLConfig(aws_token="test_token", model_name="test.model")
        
        # First call
        llm1 = config.get_llm_model()
        # Second call
        llm2 = config.get_llm_model()
        
        assert llm1 is llm2
        assert llm1 is mock_llm
        # ChatBedrockConverse should only be called once due to singleton pattern
        mock_chat_bedrock.assert_called_once_with(
            model="test.model",
            client=mock_client,
            temperature=0,
            max_tokens=4000,
        )
    
    @patch('ml_analysis_agent.config.ml_config.boto3.client')
    def test_client_configuration(self, mock_boto_client):
        """Test that client is configured with correct parameters."""
        config = AWSMLConfig(
            aws_token="test_token",
            aws_region="eu-west-1"
        )
        
        config.get_client()
        
        mock_boto_client.assert_called_once_with(
            "bedrock-runtime",
            region_name="eu-west-1",
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token="test_token",
        )
    
    @patch('ml_analysis_agent.config.ml_config.ChatBedrockConverse')
    @patch('ml_analysis_agent.config.ml_config.boto3.client')
    def test_llm_model_configuration(self, mock_boto_client, mock_chat_bedrock):
        """Test that LLM model is configured with correct parameters."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        config = AWSMLConfig(
            aws_token="test_token",
            model_name="custom.model"
        )
        
        config.get_llm_model()
        
        mock_chat_bedrock.assert_called_once_with(
            model="custom.model",
            client=mock_client,
            temperature=0,
            max_tokens=4000,
        )
    
    @patch('ml_analysis_agent.config.ml_config.boto3.client')
    def test_client_creation_failure(self, mock_boto_client):
        """Test handling of client creation failure."""
        mock_boto_client.side_effect = ClientError(
            error_response={'Error': {'Code': 'InvalidToken', 'Message': 'Invalid token'}},
            operation_name='CreateClient'
        )
        
        config = AWSMLConfig(aws_token="invalid_token")
        
        with pytest.raises(ClientError):
            config.get_client()
