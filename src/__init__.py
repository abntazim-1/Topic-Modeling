"""
Topic Modeling Project

A comprehensive topic modeling framework supporting multiple algorithms
including LDA, NMF, and BERTopic with full pipeline automation.

To minimize import-time warnings and heavy dependency initialization, this
package lazily exposes submodules via attribute access.
"""

__version__ = "0.1.0"
__author__ = "Topic Modeling Team"

def __getattr__(name):
    if name == "api":
        from . import api as _api
        return _api
    if name == "cli":
        from . import cli as _cli
        return _cli
    if name == "data":
        from . import data as _data
        return _data
    if name == "features":
        from . import features as _features
        return _features
    if name == "models":
        from . import models as _models
        return _models
    if name == "pipelines":
        from . import pipelines as _pipelines
        return _pipelines
    if name == "topics":
        from . import topics as _topics
        return _topics
    if name == "utils":
        from . import utils as _utils
        return _utils
    if name == "Settings":
        from .settings import Settings as _Settings
        return _Settings
    if name in {"__version__", "__author__"}:
        return globals()[name]
    raise AttributeError(f"module 'src' has no attribute {name}")

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
