"""
Topic Modeling Project

A comprehensive topic modeling framework supporting multiple algorithms
including LDA, NMF, and BERTopic with full pipeline automation.

This package provides:
- Data ingestion and preprocessing
- Feature extraction (TF-IDF, embeddings)
- Multiple topic modeling algorithms
- Model evaluation and inference
- API and CLI interfaces
- Pipeline orchestration
"""

# Import main components for easy access
try:
    from . import api
    from . import cli
    from . import data
    from . import features
    from . import models
    from . import pipelines
    from . import topics
    from . import utils
    from .settings import Settings
except ImportError:
    # Handle case where modules are not yet implemented
    api = None
    cli = None
    data = None
    features = None
    models = None
    pipelines = None
    topics = None
    utils = None
    Settings = None

__version__ = "0.1.0"
__author__ = "Topic Modeling Team"

__all__ = [
    "api",
    "cli", 
    "data",
    "features",
    "models",
    "pipelines",
    "topics",
    "utils",
    "Settings",
    "__version__",
    "__author__",
]
