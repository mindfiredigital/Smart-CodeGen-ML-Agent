"""ML Analysis Manager - Main orchestrator."""

import time
from langchain_core.messages import convert_to_messages

from .config.file_config import FileConfig
from .config.ml_config import MLConfig
from .agents.code_generator import CodeGeneratorAgent
from .agents.code_executor import CodeExecutorAgent
from .supervisor import SupervisorManager
from .file_manager import FileManager
from .utils.logger import get_ml_manager_logger

logger = get_ml_manager_logger()


class MLAnalysisManager:
    """Main manager class for orchestrating ML analysis operations."""

    def __init__(self, ml_config: MLConfig = None):
        logger.info("Initializing ML Analysis Manager")
        self.file_config = FileConfig()
        self.ml_config = ml_config
        self.file_manager = FileManager(self.file_config)

        logger.info("Setting up LLM model and agents")
        llm = self.ml_config.get_llm_model()
        self.code_generator = CodeGeneratorAgent(llm, self.file_config)
        self.code_executor = CodeExecutorAgent(llm)

        logger.info("Creating SupervisorManager")
        self.supervisor_manager = SupervisorManager(
            model=llm, agents=[self.code_generator, self.code_executor]
        )
        logger.info("ML Analysis Manager initialized successfully")

    def pretty_print_message(self, message, indent=False):
        """Pretty print a single message."""
        logger.info(f"Individual message from: {message}")
        pretty_message = message.pretty_repr(html=True)
        if not indent:
            print(pretty_message)
            return
        indented = '\n'.join('\t' + c for c in pretty_message.split('\n'))
        print(indented)

    def pretty_print_messages(self, update, last_message=False):
        """Pretty print messages from supervisor updates."""
        logger.info(f"Processing update batch with last_message={last_message}")
        is_subgraph = False
        if isinstance(update, tuple):
            ns, update = update
            if len(ns) == 0:
                return
            graph_id = ns[-1].split(':')[0]
            print(f'Update from subgraph {graph_id}:')
            print('\n')
            is_subgraph = True

        for node_name, node_update in update.items():
            update_label = f'Update from node {node_name}:'
            if is_subgraph:
                update_label = '\t' + update_label
            print(update_label)
            print('\n')

            messages = convert_to_messages(node_update['messages'])
            if last_message:
                messages = messages[-1:]

            for m in messages:
                self.pretty_print_message(m, indent=is_subgraph)
            print('\n')

    def run_analysis(self, user_query: str, verbose: bool = True):
        """Run ML analysis for a given user query."""
        logger.info(f"Starting analysis for query: {user_query}")
        logger.info(f"Q: {user_query}")
        
        current_file = self.file_config.get_current_data_file()
        if current_file and not any(
            ext in user_query.lower() for ext in ['.csv', '.xlsx', '.json', '.parquet']
        ):
            user_query = f'{user_query} using dataset at {current_file}'
            logger.debug(f"Modified query with dataset path: {user_query}")

        try:
            start = time.time()
            final_result = None
            for chunk in self.supervisor_manager.stream(
                {'messages': [{'role': 'user', 'content': user_query}]}
            ):
                if verbose:
                    self.pretty_print_messages(chunk, last_message=True)
                final_result = chunk

            # Extract and return only the final content
            if final_result:
                logger.debug("Processing final result")
                for node_name, node_update in final_result.items():
                    if 'messages' in node_update and node_update['messages']:
                        messages = convert_to_messages(node_update['messages'])
                        if messages:
                            last_message = messages[-1]
                            if hasattr(last_message, 'content'):
                                result = last_message.content
                                logger.info(f"A: {result}")
                                end = time.time()
                                duration = end - start
                                logger.info(f"Analysis completed successfully in {duration:.2f} seconds")
                                if verbose:
                                    print(f'\n⏱️ Total time taken: {duration:.2f} seconds')
                                return result

            logger.warning("No result available from analysis")
            return 'No result available.'
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            raise e

    def load_data_file(self, file_path: str) -> tuple[bool, str]:
        """Load a data file."""
        logger.info(f"Attempting to load data file: {file_path}")
        success, message = self.file_manager.validate_and_copy_data_file(file_path)
        if success:
            logger.info("Data file loaded successfully")
        else:
            logger.error(f"Failed to load data file: {message}")
        return success, message
