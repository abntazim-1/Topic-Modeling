import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Optional

LOG_DIR = Path(__file__).parents[2] / 'logs'
LOG_DIR.mkdir(exist_ok=True)
DEFAULT_LOG_FILE = LOG_DIR / 'project.log'

LOG_FORMAT = '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

_initialized = False

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, force: bool = False) -> None:
    """
    Sets up logging with both console and rotating file handlers.
    Creates logs/ folder if not existing.
    If `force` is True, reconfigures handlers even if already initialized.
    """
    global _initialized
    if _initialized and not force:
        return

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers when forcing reconfiguration
    if force:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file handler
    # Determine log file path: explicit arg -> env -> default
    log_file_path = (
        Path(os.environ.get("LOG_FILE_PATH")) if os.environ.get("LOG_FILE_PATH") else None
    )
    if log_file and isinstance(log_file, str):
        log_file_path = Path(log_file)
    if log_file_path is None:
        log_file_path = DEFAULT_LOG_FILE

    log_file_path.parent.mkdir(exist_ok=True, parents=True)

    fh = RotatingFileHandler(str(log_file_path), maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
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
