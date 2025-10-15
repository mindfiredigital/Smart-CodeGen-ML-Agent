
"""Data analyzer tool for analyzing various dataset formats."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from langchain.tools import tool
from .base import BaseTool


class DataAnalyzer(BaseTool):
    """Tool for analyzing various data file formats and their properties."""

    SUPPORTED_FORMATS = {
        '.csv': 'read_csv',
        '.xlsx': 'read_excel',
        '.xls': 'read_excel',
        '.parquet': 'read_parquet',
        '.json': 'read_json'
    }

    def __init__(self, file_config):
        super().__init__(
            name='data_analyzer',
            description='Analyze data file structure and return information for CSV, Excel, Parquet, and JSON files',
        )
        self.file_config = file_config

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        """Read data file based on its extension."""
        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {extension}")
        
        read_method = getattr(pd, self.SUPPORTED_FORMATS[extension])
        
        try:
            if extension in ['.xlsx', '.xls']:
                return read_method(file_path, engine='openpyxl')
            return read_method(file_path)
        except Exception as e:
            raise RuntimeError(f"Error reading {extension} file: {str(e)}")

    def _analyze_dataframe(self, df: pd.DataFrame, file_path: Path) -> dict:
        """Analyze a pandas DataFrame and return its properties."""
        try:
            analysis = {
                'file_path': str(file_path),
                'file_type': file_path.suffix.lower(),
                'shape': [int(df.shape[0]), int(df.shape[1])],
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': {col: int(count) for col, count in df.isnull().sum().items()},
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'unique_values': {col: int(df[col].nunique()) for col in df.columns},
            }

            # Add basic stats for numeric columns if they exist
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                analysis['basic_stats'] = {
                    col: {stat: float(val) for stat, val in stats.items()}
                    for col, stats in numeric_df.describe().to_dict().items()
                }

            # Add sample data (first few rows)
            analysis['sample_data'] = json.loads(df.head(3).to_json(orient='records'))

            return analysis

        except Exception as e:
            raise RuntimeError(f"Error analyzing DataFrame: {str(e)}")

    def execute(self, file_path: str) -> str:
        """Analyze data file structure and return detailed information."""
        try:
            path = Path(file_path)
            if not path.exists():
                path = self.file_config.get_data_path(file_path)
                if not path.exists():
                    return self.format_failure(f'File not found: {file_path}')

            # Read the data file
            df = self._read_file(path)
            
            # Analyze the DataFrame
            analysis = self._analyze_dataframe(df, path)
            
            return f'✅ Data Analysis Complete:\n{json.dumps(analysis, indent=2)}'

        except Exception as e:
            return self.handle_error(e)


@tool('data_analyzer', return_direct=False)
def data_analyzer_tool(file_path: str) -> str:
    """Analyze data file structure for CSV, Excel, Parquet, or JSON files."""
    from ..config.file_config import FileConfig

    file_config = FileConfig()
    analyzer = DataAnalyzer(file_config)
    return analyzer.execute(file_path)