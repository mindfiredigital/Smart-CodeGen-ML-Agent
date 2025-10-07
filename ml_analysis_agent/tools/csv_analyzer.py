"""CSV analyzer tool for analyzing dataset structure."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from langchain.tools import tool
from .base import BaseTool


class CSVAnalyzer(BaseTool):
    """Tool for analyzing CSV file structure and properties."""

    def __init__(self, file_config):
        super().__init__(
            name='csv_analyzer',
            description='Analyze CSV file structure and return column information',
        )
        self.file_config = file_config

    def execute(self, csv_file_path: str) -> str:
        """Analyze CSV file structure and return column information."""
        try:
            file_path = Path(csv_file_path)
            if not file_path.exists():
                file_path = self.file_config.get_data_path(csv_file_path)
                if not file_path.exists():
                    return self.format_failure(f'CSV file not found: {csv_file_path}')

            df = pd.read_csv(file_path)

            analysis = {
                'file_path': str(file_path),
                'shape': [int(df.shape[0]), int(df.shape[1])],
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': {col: int(count) for col, count in df.isnull().sum().items()},
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'unique_values': {col: int(df[col].nunique()) for col in df.columns},
                'basic_stats': {
                    col: {stat: float(val) for stat, val in stats.items()}
                    for col, stats in df.describe().to_dict().items()
                }
                if len(df.select_dtypes(include=[np.number]).columns) > 0
                else {},
            }

            return f'✅ CSV Analysis Complete:\n{json.dumps(analysis, indent=2)}'

        except Exception as e:
            return self.handle_error(e)


@tool('csv_analyzer', return_direct=False)
def csv_analyzer_tool(csv_file_path: str) -> str:
    """Analyze CSV file structure."""
    from ..config.file_config import FileConfig

    file_config = FileConfig()
    analyzer = CSVAnalyzer(file_config)
    return analyzer.execute(csv_file_path)
