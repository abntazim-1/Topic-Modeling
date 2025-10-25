#!/usr/bin/env python3
"""
CLI Entry Point for Topic Modeling Project
==========================================

This script provides the main entry point for the command-line interface.
It can be run directly or through the module system.

Usage:
    python cli.py train --model lda --topics 10
    python -m src.cli.cli train --model lda --topics 10
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import and run the CLI
from src.cli.cli import cli

if __name__ == '__main__':
    cli()
