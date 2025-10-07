"""File configuration settings for the ML Analysis System."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class FileConfig:
    """Configuration class for file paths and directories."""

    def __init__(self):
        self.OUTPUT_DIR = Path('temp')
        self.CSV_DATA_DIR = Path('data')
        self.CURRENT_DATA_FILE = None

        # Create directories if they don't exist
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CSV_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Supported file extensions
        self.VALID_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.json', '.parquet']

    def get_output_path(self, filename: str = 'ml_analysis.py') -> Path:
        """Get the output file path."""
        return self.OUTPUT_DIR / filename

    def get_data_path(self, filename: str = None) -> Path:
        """Get the data directory path or specific file path."""
        if filename:
            return self.CSV_DATA_DIR / filename
        return self.CSV_DATA_DIR

    def set_current_data_file(self, file_path: str):
        """Set the current data file path."""
        self.CURRENT_DATA_FILE = file_path

    def get_current_data_file(self) -> str:
        """Get the current data file path."""
        return self.CURRENT_DATA_FILE
