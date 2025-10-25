"""
API Routes for Topic Modeling
============================

Additional route definitions and utilities for the Topic Modeling API.
This module provides route decorators and utility functions.
"""

from functools import wraps
from flask import request, jsonify
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_json_content(f):
    """Decorator to validate JSON content in requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        return f(*args, **kwargs)
    return decorated_function


def validate_model_type(f):
    """Decorator to validate model type parameter."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        model_type = kwargs.get('model_type')
        if model_type and model_type not in ['lda', 'nmf']:
            return jsonify({'error': 'Invalid model type. Use "lda" or "nmf"'}), 400
        return f(*args, **kwargs)
    return decorated_function


def log_request(f):
    """Decorator to log API requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger.info(f"API Request: {request.method} {request.path}")
        logger.debug(f"Request data: {request.get_json()}")
        return f(*args, **kwargs)
    return decorated_function


def handle_api_errors(f):
    """Decorator to handle API errors gracefully."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"API Error in {f.__name__}: {e}")
            return jsonify({'error': 'Internal server error', 'message': str(e)}), 500
    return decorated_function
