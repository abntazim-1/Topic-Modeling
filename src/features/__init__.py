"""
Features module for the Topic Modeling project.

This module handles feature extraction including embeddings and TF-IDF vectors.
"""

# Import main components for easy access
try:
    from .embeddings import EmbeddingExtractor
    from .tfidf import TFIDFExtractor
except ImportError:
    # Handle case where modules are not yet implemented
    EmbeddingExtractor = None
    TFIDFExtractor = None

__all__ = [
    "EmbeddingExtractor",
    "TFIDFExtractor",
]
