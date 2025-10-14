"""
Models module for the Topic Modeling project.

This module handles model training, evaluation, and inference.
"""

# Import main components for easy access
try:
    from .train import ModelTrainer
    from .evaluate import ModelEvaluator
    from .infer import ModelInference
    from .utils import ModelUtils
except ImportError:
    # Handle case where modules are not yet implemented
    ModelTrainer = None
    ModelEvaluator = None
    ModelInference = None
    ModelUtils = None

__all__ = [
    "ModelTrainer",
    "ModelEvaluator",
    "ModelInference", 
    "ModelUtils",
]
