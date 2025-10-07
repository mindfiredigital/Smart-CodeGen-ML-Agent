"""Prompt manager for loading and managing system prompts."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class PromptManager:
    """Manager class for loading and formatting prompts from YAML configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the PromptManager.

        Args:
            config_path: Path to the prompts.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Default to prompts.yaml in the config directory
            config_path = Path(__file__).parent / 'prompts.yaml'

        self.config_path = Path(config_path)
        self._prompts_data = None
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._prompts_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'Prompts configuration file not found: {self.config_path}')
        except yaml.YAMLError as e:
            raise ValueError(f'Error parsing prompts YAML file: {str(e)}')

    def get_prompt(self, agent_name: str, **kwargs) -> str:
        """
        Get a prompt for a specific agent with optional formatting.

        Args:
            agent_name: Name of the agent (e.g., 'code_generator', 'code_executor', 'supervisor')
            **kwargs: Keyword arguments for formatting the prompt template

        Returns:
            Formatted prompt string
        """
        if agent_name not in self._prompts_data:
            raise ValueError(f"Agent '{agent_name}' not found in prompts configuration")

        prompt_template = self._prompts_data[agent_name].get('system_prompt', '')

        # Format the prompt with provided kwargs
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f'Missing required parameter for prompt formatting: {str(e)}')

    def get_agent_info(self, agent_name: str) -> Dict[str, str]:
        """
        Get agent information (name and description).

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with 'name' and 'description' keys
        """
        if agent_name not in self._prompts_data:
            raise ValueError(f"Agent '{agent_name}' not found in prompts configuration")

        agent_data = self._prompts_data[agent_name]
        return {
            'name': agent_data.get('name', agent_name),
            'description': agent_data.get('description', ''),
        }

    def get_template(self, template_name: str, **kwargs) -> str:
        """
        Get a template from the templates section.

        Args:
            template_name: Name of the template
            **kwargs: Keyword arguments for formatting the template

        Returns:
            Formatted template string
        """
        templates = self._prompts_data.get('templates', {})
        if template_name not in templates:
            raise ValueError(f"Template '{template_name}' not found in prompts configuration")

        template = templates[template_name]
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f'Missing required parameter for template formatting: {str(e)}')

    def get_error_message(self, error_key: str, **kwargs) -> str:
        """
        Get a formatted error message.

        Args:
            error_key: Key for the error message
            **kwargs: Keyword arguments for formatting the error message

        Returns:
            Formatted error message string
        """
        error_messages = self._prompts_data.get('templates', {}).get('error_messages', {})
        if error_key not in error_messages:
            return f'❌ Error: {error_key}'

        error_template = error_messages[error_key]
        try:
            return error_template.format(**kwargs)
        except KeyError:
            return error_template

    def get_setting(self, setting_key: str, default: Any = None) -> Any:
        """
        Get a system-wide setting.

        Args:
            setting_key: Key for the setting
            default: Default value if setting not found

        Returns:
            Setting value or default
        """
        settings = self._prompts_data.get('settings', {})
        return settings.get(setting_key, default)

    def reload(self):
        """Reload prompts from the YAML file."""
        self._load_prompts()

    def list_agents(self) -> list:
        """
        Get list of available agent names.

        Returns:
            List of agent names
        """
        return [key for key in self._prompts_data.keys() if key not in ['templates', 'settings']]

    def list_templates(self) -> list:
        """
        Get list of available template names.

        Returns:
            List of template names
        """
        templates = self._prompts_data.get('templates', {})
        return [key for key in templates.keys() if key != 'error_messages']

    def list_error_messages(self) -> list:
        """
        Get list of available error message keys.

        Returns:
            List of error message keys
        """
        error_messages = self._prompts_data.get('templates', {}).get('error_messages', {})
        return list(error_messages.keys())


# Singleton instance for easy access
_prompt_manager_instance = None


def get_prompt_manager(config_path: Optional[str] = None) -> PromptManager:
    """
    Get or create a singleton instance of PromptManager.

    Args:
        config_path: Path to the prompts.yaml file

    Returns:
        PromptManager instance
    """
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager(config_path)
    return _prompt_manager_instance
