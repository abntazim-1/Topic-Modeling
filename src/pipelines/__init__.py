"""
Pipelines module for the Topic Modeling project.

This module orchestrates the complete topic modeling pipeline
from data ingestion to model deployment.
"""

# Import main components for easy access
try:
    from .run_pipeline import TopicModelingPipeline
except ImportError:
    # Handle case where modules are not yet implemented
    TopicModelingPipeline = None

__all__ = [
    "TopicModelingPipeline",
]
