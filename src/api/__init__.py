"""
API module for the Topic Modeling project.

This module contains the FastAPI application and route definitions
for serving topic modeling endpoints.
"""

# Import main components for easy access
try:
    from .app import app
    from .routes import router
except ImportError:
    # Handle case where modules are not yet implemented
    app = None
    router = None

__all__ = [
    "app",
    "router",
]
