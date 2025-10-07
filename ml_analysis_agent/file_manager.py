"""File manager for handling data file operations."""

import shutil
import atexit
from pathlib import Path
from typing import Tuple

from .config.prompt_manager import get_prompt_manager
from .utils.logger import get_file_manager_logger

logger = get_file_manager_logger()


class FileManager:
    """Manager class for file operations and data handling."""

    def __init__(self, file_config):
        self.file_config = file_config
        self.prompt_manager = get_prompt_manager()
        self._register_cleanup()

    def _register_cleanup(self):
        """Register cleanup function to run on program exit."""
        atexit.register(self.cleanup_data_folder)

    def cleanup_data_folder(self):
        """Remove the entire data folder on program exit."""
        try:
            if self.file_config.CSV_DATA_DIR.exists():
                logger.info(f"Cleaning up data folder: {self.file_config.CSV_DATA_DIR}")
                shutil.rmtree(self.file_config.CSV_DATA_DIR)
                logger.info("Data folder cleanup successful")
        except Exception as e:
            error_msg = f"Could not remove data folder: {str(e)}"
            logger.error(error_msg, exc_info=True)

    def validate_file_extension(self, file_path: Path) -> bool:
        """Validate if file has a supported extension."""
        return file_path.suffix.lower() in self.file_config.VALID_EXTENSIONS

    def validate_and_copy_data_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate and copy data file to the data directory."""
        logger.info(f"Validating and copying data file: {file_path}")
        try:
            source_path = Path(file_path)

            if not source_path.exists():
                error_msg = self.prompt_manager.get_error_message(
                    'file_not_found', file_path=file_path
                )
                logger.error(f"File not found: {file_path}")
                return False, error_msg

            if not self.validate_file_extension(source_path):
                valid_exts = ', '.join(self.file_config.VALID_EXTENSIONS)
                error_msg = self.prompt_manager.get_error_message(
                    'unsupported_format', valid_extensions=valid_exts
                )
                logger.error(f"Unsupported file format: {source_path.suffix}. Valid extensions: {valid_exts}")
                return False, error_msg

            dest_path = self.file_config.get_data_path(f'current_data{source_path.suffix}')
            logger.debug(f"Copying file from {source_path} to {dest_path}")
            shutil.copy2(source_path, dest_path)

            self.file_config.set_current_data_file(str(dest_path))
            logger.info(f"Data file loaded successfully: {source_path.name}")

            return True, f'✅ Data file loaded successfully: {source_path.name}'

        except Exception as e:
            error_msg = f"Error loading data file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, f'❌ {error_msg}'
