"""
Data module for the Topic Modeling project.

This module handles data ingestion, preprocessing, and dataset management.
"""

# Import main components for easy access
try:
    from .dataset import Dataset
    from .ingest import DataIngestion
    from .preprocess import TextPreprocessor
except ImportError:
    # Handle case where modules are not yet implemented
    Dataset = None
    DataIngestion = None
    TextPreprocessor = None

__all__ = [
    "Dataset",
    "DataIngestion", 
    "TextPreprocessor",
]
