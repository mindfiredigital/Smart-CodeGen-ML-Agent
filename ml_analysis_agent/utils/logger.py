"""Logging configuration for ML Analysis Agent."""

import logging
import os
import time
from datetime import datetime
from typing import Optional

# Define a custom logging level for Q&A messages
QA_LEVEL_NUM = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(QA_LEVEL_NUM, 'QA')

# Add qa method to Logger class
def qa(self, message, *args, **kwargs):
    """Log Q&A related messages at QA level."""
    if self.isEnabledFor(QA_LEVEL_NUM):
        self._log(QA_LEVEL_NUM, message, args, **kwargs)

logging.Logger.qa = qa

class QAFilter(logging.Filter):
    """Filter that only allows QA level messages to pass."""
    def filter(self, record):
        return record.levelno == QA_LEVEL_NUM

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    """Set up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file. If None, a timestamped file will be created
        level: Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get the project root directory (where the logs folder should be)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(project_root, "logs")
    
    # Ensure the logs directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Set up logging format with milliseconds
    logging.Formatter.converter = time.gmtime  # Use UTC time
    
    # All logs will go to main.log
    main_log_file = os.path.join(log_dir, "main.log")
    
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # File handler for all levels going to main.log
        file_handler = logging.FileHandler(main_log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Add console handler for specific loggers
        if name == 'cli':
            # CLI logger gets ERROR level messages
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        elif name == 'ml_manager':
            # ML Manager gets only QA level messages in console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(QA_LEVEL_NUM)
            console_handler.addFilter(QAFilter())
            # Use simpler format for QA messages
            qa_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(qa_formatter)
            logger.addHandler(console_handler)

    return logger

def get_agent_logger(agent_name: str):
    """Get logger for an agent module."""
    return setup_logger(f"agent.{agent_name}")

def get_tool_logger(tool_name: str):
    """Get logger for a tool module."""
    return setup_logger(f"tool.{tool_name}")

def get_cli_logger():
    """Get logger for CLI operations."""
    return setup_logger("cli")

def get_main_logger():
    """Get logger for main operations."""
    return setup_logger("main")

def get_config_logger():
    """Get logger for configuration operations."""
    return setup_logger("config")

def get_file_manager_logger():
    """Get logger for file manager operations."""
    return setup_logger("file_manager")

def get_ml_manager_logger():
    """Get logger for ML manager operations."""
    return setup_logger("ml_manager")