"""
Safe Code Executor for LLM-generated Python code.
Executes pandas/matplotlib/seaborn code and captures visualizations.
"""

import io
import sys
import base64
import logging
import traceback
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Allowed modules for code execution
ALLOWED_MODULES = {
    'pandas', 'numpy', 'matplotlib', 'matplotlib.pyplot', 'seaborn',
    'scipy', 'scipy.stats', 'math', 'statistics'
}

# Forbidden patterns in code
FORBIDDEN_PATTERNS = [
    'import os', 'import sys', 'import subprocess', 'import shutil',
    '__import__', 'eval(', 'exec(', 'compile(', 'open(',
    'file(', 'input(', 'raw_input(',
    'os.system', 'os.popen', 'subprocess.', 'shutil.',
    '.read(', '.write(', '.delete', '.remove', '.unlink',
    'pickle', 'shelve', 'socket', 'urllib', 'requests',
    'globals(', 'locals(', 'vars(', 'dir(',
    '__builtins__', '__code__', '__class__'
]


class CodeExecutor:
    def __init__(self):
        self.figures = []
        
    def validate_code(self, code: str) -> tuple[bool, str]:
        """Validate code for safety before execution."""
        code_lower = code.lower()
        
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.lower() in code_lower:
                return False, f"Forbidden pattern detected: {pattern}"
        
        return True, "Code validated"
    
    def execute(self, code: str, df, df_name: str = 'df') -> dict:
        """
        Execute Python code with the given DataFrame.
        
        Args:
            code: Python code to execute
            df: pandas DataFrame to make available
            df_name: Name to use for the DataFrame in code
            
        Returns:
            dict with 'success', 'figures', 'output', 'error'
        """
        # Validate code first
        is_valid, validation_msg = self.validate_code(code)
        if not is_valid:
            return {
                'success': False,
                'figures': [],
                'output': '',
                'error': validation_msg
            }
        
        # Prepare execution environment
        import pandas as pd
        import numpy as np
        import seaborn as sns
        
        # Close any existing figures
        plt.close('all')
        
        # Create safe globals
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
                'map': map,
                'filter': filter,
                'any': any,
                'all': all,
                'isinstance': isinstance,
                'type': type,
            },
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'matplotlib': matplotlib,
            'io': io,
            'base64': base64,
            df_name: df.copy(),  # Use a copy for safety
        }
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        figures = []
        error = None
        
        try:
            # Execute the code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals)
            
            # Capture all figures
            fig_nums = plt.get_fignums()
            for fig_num in fig_nums:
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                           facecolor='white', edgecolor='none')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                figures.append(img_base64)
                buf.close()
            
            plt.close('all')
            
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Code execution error: {error}")
        
        output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        if stderr_output and not error:
            error = stderr_output
        
        return {
            'success': error is None,
            'figures': figures,
            'output': output,
            'error': error
        }


def execute_visualization_code(code: str, df, df_name: str = 'df') -> dict:
    """
    Convenience function to execute visualization code.
    
    Args:
        code: Python code to execute
        df: pandas DataFrame
        df_name: Variable name for DataFrame in code
        
    Returns:
        dict with execution results
    """
    executor = CodeExecutor()
    return executor.execute(code, df, df_name)
