"""
CLI module for the Topic Modeling project.

This module provides command-line interface functionality
for running topic modeling pipelines and operations.
"""

# Import main components for easy access
try:
    from .cli import cli_app
except ImportError:
    # Handle case where modules are not yet implemented
    cli_app = None

__all__ = [
    "cli_app",
]
