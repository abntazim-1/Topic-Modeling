import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Optional

LOG_DIR = Path(__file__).parents[2] / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'project.log'

LOG_FORMAT = '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

_initialized = False

def setup_logging(log_level: str = "INFO") -> None:
    """
    Sets up production-grade logging with both console and rotating file handlers.
    Creates logs/ folder if not existing. Should be called once (idempotent).
    """
    global _initialized
    if _initialized:
        return

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    fh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    _initialized = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a logger instance with the given name. Ensures logging is configured.
    Usage: logger = get_logger(__name__)
    """
    setup_logging()
    return logging.getLogger(name)
