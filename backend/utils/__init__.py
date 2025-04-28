# utils/__init__.py
"""
Utility modules for data processing and analysis.
"""

import logging

# Configure logging for utility modules
logger = logging.getLogger('utils')
logger.setLevel(logging.INFO)

# Define version
__version__ = '1.0.0'