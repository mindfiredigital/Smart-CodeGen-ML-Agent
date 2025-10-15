"""ML and AWS configuration settings."""
import os
import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()

class MLConfig:
    """Base configuration class."""

    def get_llm_model(self):
        """Method to get LLM model instance."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def get_client(self):
        """Method to get client instance."""
        pass

class AWSMLConfig(MLConfig):
    """Configuration class for ML and AWS settings."""
    
    def __init__(
            self, 
            aws_token: str = None, 
            aws_region: str = None, 
            model_name: str = None
        ):
        self.AWS_BEARER_TOKEN_BEDROCK = aws_token or os.getenv('AWS_BEARER_TOKEN_BEDROCK')
        self.AWS_DEFAULT_REGION = aws_region or os.getenv('AWS_DEFAULT_REGION', 'us-west-2')
        self.MODEL_NAME = model_name or os.getenv("MODEL_NAME","us.anthropic.claude-sonnet-4-20250514-v1:0")
        
        if not self.AWS_BEARER_TOKEN_BEDROCK:
            raise ValueError("AWS_BEARER_TOKEN_BEDROCK not found in .env file.")
        
        self._client = None
        self._llm = None
    
    def get_client(self):
        """Get AWS Bedrock client (singleton pattern)."""
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.AWS_DEFAULT_REGION,
                aws_access_key_id=None,
                aws_secret_access_key=None,
                aws_session_token=self.AWS_BEARER_TOKEN_BEDROCK,
            )
        return self._client
    
    def get_llm_model(self):
        """Get LLM model instance (singleton pattern)."""
        if self._llm is None:
            self._llm = ChatBedrockConverse(
                model=self.MODEL_NAME,
                client=self.get_client(),
                temperature=0,
                max_tokens=4000,
            )
        return self._llm
from langchain_openai import ChatOpenAI

class OpenAIMLConfig(MLConfig):
    """Configuration class for OpenAI settings."""
    
    def __init__(
            self,
            api_key: str = None,
            model_name: str = None,
            temperature: float = 0,
            max_tokens: int = 4000
        ):
        self.OPENAI_API_KEY = api_key or os.getenv('OPENAI_API_KEY')
        self.MODEL_NAME = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo-preview")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        
        self._llm = None
    
    def get_llm_model(self):
        """Get OpenAI LLM model instance (singleton pattern)."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.MODEL_NAME,
                openai_api_key=self.OPENAI_API_KEY,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._llm

