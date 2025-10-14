"""
Topics module for the Topic Modeling project.

This module contains implementations of different topic modeling algorithms
including LDA, NMF, and BERTopic.
"""

# Import main components for easy access
try:
    from .lda_model import LDAModel
    from .nmf_model import NMFModel
    from .bertopic_model import BERTopicModel
except ImportError:
    # Handle case where modules are not yet implemented
    LDAModel = None
    NMFModel = None
    BERTopicModel = None

__all__ = [
    "LDAModel",
    "NMFModel",
    "BERTopicModel",
]
