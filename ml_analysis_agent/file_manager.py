"""File manager for handling data file operations."""

import shutil
import atexit
from pathlib import Path
from typing import Tuple
from .config.prompt_manager import get_prompt_manager


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
                shutil.rmtree(self.file_config.CSV_DATA_DIR)
                print(f'🧹 Cleaned up data folder: {self.file_config.CSV_DATA_DIR}')
        except Exception as e:
            print(f'⚠️ Could not remove data folder: {str(e)}')

    def validate_file_extension(self, file_path: Path) -> bool:
        """Validate if file has a supported extension."""
        return file_path.suffix.lower() in self.file_config.VALID_EXTENSIONS

    def validate_and_copy_data_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate and copy data file to the data directory."""
        try:
            source_path = Path(file_path)

            if not source_path.exists():
                error_msg = self.prompt_manager.get_error_message(
                    'file_not_found', file_path=file_path
                )
                return False, error_msg

            if not self.validate_file_extension(source_path):
                valid_exts = ', '.join(self.file_config.VALID_EXTENSIONS)
                error_msg = self.prompt_manager.get_error_message(
                    'unsupported_format', valid_extensions=valid_exts
                )
                return False, error_msg

            dest_path = self.file_config.get_data_path(f'current_data{source_path.suffix}')
            shutil.copy2(source_path, dest_path)

            self.file_config.set_current_data_file(str(dest_path))

            return True, f'✅ Data file loaded successfully: {source_path.name}'

        except Exception as e:
            return False, f'❌ Error loading data file: {str(e)}'
