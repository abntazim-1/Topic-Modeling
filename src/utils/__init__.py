"""
Utils module for the Topic Modeling project.

This module provides utility functions for I/O operations,
logging, and text processing.
"""

# Import main components for easy access
try:
    from .io import FileIO, DataIO
    from .logging_utils import setup_logging, get_logger
    from .text import TextProcessor, TextCleaner
except ImportError:
    # Handle case where modules are not yet implemented
    FileIO = None
    DataIO = None
    setup_logging = None
    get_logger = None
    TextProcessor = None
    TextCleaner = None

__all__ = [
    "FileIO",
    "DataIO",
    "setup_logging",
    "get_logger",
    "TextProcessor",
    "TextCleaner",
]
