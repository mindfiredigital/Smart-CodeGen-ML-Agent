"""Utility functions for handling user input."""


def get_user_input():
    """Get AWS configuration from user input.

    Returns:
        tuple: (token, region, model_id)
            - token (str): AWS Bedrock token
            - region (str): AWS region (defaults to us-west-2)
            - model_id (str): Model ID (defaults to anthropic.claude-v2)
    """
    token = input('Enter your AWS Bedrock token: ').strip()
    region = input('Enter AWS region (default: us-west-2): ').strip() or 'us-west-2'
    model_id = (
        input('Enter model ID (default: anthropic.claude-v2): ').strip() or 'anthropic.claude-v2'
    )
    return token, region, model_id
