"""Code executor tool for running generated Python scripts."""

import sys
import subprocess
import re
import io
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from langchain.tools import tool
from .base import BaseTool


class CodeExecutor(BaseTool):
    """Tool for executing generated Python code."""

    # Package name mappings for deprecated or aliased packages
    PACKAGE_MAPPINGS = {
        'sklearn': 'scikit-learn',
        'PIL': 'pillow',
    }

    def __init__(self, file_config):
        super().__init__(
            name='code_executor_tool', description='Execute a generated ML Python script'
        )
        self.file_config = file_config

    def install_dependency(self, package: str) -> str:
        """Install a package if it's not already installed."""
        try:
            __import__(package)
            return f"✅ Dependency '{package}' already installed."
        except ImportError:
            try:
                # Use the correct package name if it's in our mapping
                install_name = self.PACKAGE_MAPPINGS.get(package, package)
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', install_name])
                return f'📦 Installed missing dependency: {install_name}'
            except Exception as e:
                return f'❌ Failed to install {package}: {str(e)}'

    def check_and_install_dependencies(self, file_path: Path):
        """Check code for imports and install missing dependencies."""
        code = file_path.read_text()
        imports = re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_]+)', code, re.MULTILINE)
        logs = []
        for pkg in set(imports):
            logs.append(self.install_dependency(pkg))
        return '\n'.join(logs), code

    def execute(self, file_path: str) -> str:
        """Execute a generated ML Python script."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return self.format_failure(f'No such file: {file_path}')

            dep_logs, code_content = self.check_and_install_dependencies(file_path)

            exec_globals = {
                '__name__': '__main__',
                '__file__': str(file_path),
            }

            output_buffer = io.StringIO()
            error_buffer = io.StringIO()

            try:
                with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                    exec(code_content, exec_globals)

                stdout_content = output_buffer.getvalue()
                stderr_content = error_buffer.getvalue()

                if stdout_content and stderr_content:
                    return f'{stdout_content}\n⚠️ {stderr_content}'
                elif stdout_content:
                    return stdout_content
                elif stderr_content:
                    return f'⚠️ {stderr_content}'
                else:
                    return '✅ Code executed but produced no output'

            except Exception as exec_error:
                return self.format_failure(f'Execution Error: {exec_error}')

        except Exception as e:
            return self.handle_error(e)


@tool
def code_executor_tool(file_path: str) -> str:
    """Execute a generated ML Python script."""
    from ..config.file_config import FileConfig

    file_config = FileConfig()
    executor = CodeExecutor(file_config)
    return executor.execute(file_path)
