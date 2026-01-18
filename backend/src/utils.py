"""
utils.py
--------
Shared utility functions for the NanoSentri pipeline.
Handles logging configuration to ensure consistent output formats across
Cloud (Colab) and Edge (Local) environments.
"""

import logging
import sys
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a standardized logger.
    
    Args:
        name (str): The name of the module calling the logger.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler with specific format
        # Format: [Time] [Level] [Module]: Message
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    Assumes this file is located in src/
    """
    return Path(__file__).parent.parent.resolve()