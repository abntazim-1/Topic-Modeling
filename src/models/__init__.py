"""
Models module for the Topic Modeling project.

This module handles model training, evaluation, and inference.

To reduce import-time side effects and noisy warnings from heavy dependencies,
we lazily import components only when they are accessed.
"""

# Lazy attribute loader to avoid importing heavy modules at package import time
def __getattr__(name):
    if name == "ModelTrainer":
        from .train import ModelTrainer as _ModelTrainer
        return _ModelTrainer
    if name == "ModelEvaluator":
        from .evaluate import ModelEvaluator as _ModelEvaluator
        return _ModelEvaluator
    if name == "ModelInference":
        from .infer import ModelInference as _ModelInference
        return _ModelInference
    if name == "ModelUtils":
        from .utils import ModelUtils as _ModelUtils
        return _ModelUtils
    raise AttributeError(f"module 'src.models' has no attribute {name}")

__all__ = [
    "ModelTrainer",
    "ModelEvaluator",
    "ModelInference",
    "ModelUtils",
]
