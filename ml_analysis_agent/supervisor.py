"""Supervisor for managing multiple agents."""

from typing import List
from langgraph_supervisor import create_supervisor
from IPython.display import display, Image
from .config.prompt_manager import get_prompt_manager
from .utils.logger import get_main_logger

logger = get_main_logger()


class SupervisorManager:
    """Manager class for the supervisor that coordinates multiple agents."""

    def __init__(self, model, agents: List):
        logger.info("Initializing SupervisorManager")
        self.model = model
        logger.debug(f"Initializing {len(agents)} agents")
        self.agents = [agent.create_agent() for agent in agents]
        self.prompt_manager = get_prompt_manager()
        self._supervisor = None
        self.graph_path = None
        logger.info("SupervisorManager initialized successfully")

    def get_supervisor_prompt(self) -> str:
        """Get the supervisor prompt."""
        # Use PromptManager to get the prompt
        return self.prompt_manager.get_prompt('supervisor')

    def create_supervisor(self, save_graph=True):
        """Create the supervisor using langgraph.
        
        Args:
            save_graph (bool): Whether to save and display the supervisor graph. Defaults to True.
        """
        if self._supervisor is None:
            logger.info("Creating new supervisor instance")
            try:
                prompt = self.get_supervisor_prompt()
                logger.debug("Supervisor prompt retrieved")
                
                self._supervisor = create_supervisor(
                    model=self.model,
                    agents=self.agents,
                    prompt=prompt,
                    add_handoff_back_messages=True,
                    output_mode='full_history',
                ).compile()
                logger.info("Supervisor created and compiled successfully")
                
                # Generate and save the supervisor graph
                if save_graph:
                    try:
                        from pathlib import Path
                        logger.info("Generating supervisor graph visualization")
                        png_bytes = self._supervisor.get_graph().draw_mermaid_png()
                        
                        # Create graphs directory if it doesn't exist
                        graphs_dir = Path("graphs")
                        graphs_dir.mkdir(exist_ok=True)
                        
                        # Save the graph file
                        self.graph_path = graphs_dir / "supervisor_graph.png"
                        with open(self.graph_path, "wb") as f:
                            f.write(png_bytes)
                        logger.info(f"Supervisor graph saved to {self.graph_path}")
                    except Exception as e:
                        logger.warning(f"Failed to generate supervisor graph: {str(e)}")
                        self.graph_path = None
            except Exception as e:
                logger.error(f"Failed to create supervisor: {str(e)}", exc_info=True)
                raise
        return self._supervisor

    def invoke(self, input_data: dict):
        """Invoke the supervisor with input data."""
        logger.info("Invoking supervisor")
        logger.debug(f"Input data: {input_data}")
        try:
            supervisor = self.create_supervisor()
            result = supervisor.invoke(input_data)
            logger.info("Supervisor invocation completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error during supervisor invocation: {str(e)}", exc_info=True)
            raise

    def stream(self, input_data: dict):
        """Stream responses from the supervisor."""
        logger.info("Starting supervisor stream")
        logger.debug(f"Input data: {input_data}")
        try:
            supervisor = self.create_supervisor()
            for chunk in supervisor.stream(input_data):
                logger.debug(f"Streaming chunk: {chunk}")
                yield chunk
            logger.info("Supervisor stream completed successfully")
        except Exception as e:
            logger.error(f"Error during supervisor stream: {str(e)}", exc_info=True)
            raise
            
