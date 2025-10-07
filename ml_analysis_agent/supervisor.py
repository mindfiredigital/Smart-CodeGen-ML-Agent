"""Supervisor for managing multiple agents."""

from typing import List
from langgraph_supervisor import create_supervisor
from .config.prompt_manager import get_prompt_manager


class SupervisorManager:
    """Manager class for the supervisor that coordinates multiple agents."""

    def __init__(self, model, agents: List):
        self.model = model
        self.agents = [agent.create_agent() for agent in agents]
        self.prompt_manager = get_prompt_manager()
        self._supervisor = None

    def get_supervisor_prompt(self) -> str:
        """Get the supervisor prompt."""
        # Use PromptManager to get the prompt
        return self.prompt_manager.get_prompt('supervisor')

    def create_supervisor(self):
        """Create the supervisor using langgraph."""
        if self._supervisor is None:
            self._supervisor = create_supervisor(
                model=self.model,
                agents=self.agents,
                prompt=self.get_supervisor_prompt(),
                add_handoff_back_messages=True,
                output_mode='full_history',
            ).compile()
        return self._supervisor

    def invoke(self, input_data: dict):
        """Invoke the supervisor with input data."""
        supervisor = self.create_supervisor()
        return supervisor.invoke(input_data)

    def stream(self, input_data: dict):
        """Stream responses from the supervisor."""
        supervisor = self.create_supervisor()
        return supervisor.stream(input_data)
