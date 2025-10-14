import traceback
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AppException(Exception):
    """
    Base custom exception that logs errors and full traceback on instantiation.
    """
    def __init__(self, message: str):
        super().__init__(message)
        tb = traceback.format_exc()
        # Only log traceback if currently handling an exception
        if tb and tb != 'NoneType: None\n':
            logger.error(f"{self.__class__.__name__}: {message}\nTraceback:\n{tb}")
        else:
            logger.error(f"{self.__class__.__name__}: {message}")

class DataValidationError(AppException):
    """
    Raised when input data validation fails.
    """
    pass

class ModelInferenceError(AppException):
    """
    Raised when model inference or prediction fails.
    """
    pass
