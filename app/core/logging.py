import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from app.core.config import get_settings


settings = get_settings()  # Load settings to access debug flag

# Set up logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Log Format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> logging.Logger:
    """ Set up logging for the application.
        Logs are written to both console and file
    """

    # Log level
    log_level = logging.DEBUG if settings.debug else logging.INFO

    # Root logger
    logger = logging.getLogger(settings.app_name)
    logger.setLevel(log_level)

    # avoid adding multiple handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger

    # Formatter
    logger_formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    consoler_handler = logging.StreamHandler(sys.stdout)
    consoler_handler.setLevel(log_level)
    consoler_handler.setFormatter(logger_formatter)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        LOG_DIR/"app.log",
        maxBytes=5 * 1024 * 1024, # 5 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logger_formatter)

    # Error file handler with rotation
    error_file_handler = RotatingFileHandler(
        LOG_DIR/"error.log",
        maxBytes=5 * 1024 * 1024, # 5 MB
        backupCount=5,
        encoding="utf-8"
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(logger_formatter)

    # Add handlers to logger
    logger.addHandler(consoler_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)

    logger.info("=" * 60)
    logger.info(f"{settings.app_name} v {settings.api_version}")
    logger.info(f" Debug mode: {settings.debug}")
    logger.info(f" Log level: {logging.getLevelName(log_level)}")
    logger.info(f" Log Directory: {LOG_DIR.resolve()}")
    logger.info("=" * 60)

    return logger

def get_logger(name: str) -> logging.Logger:
    """ Get a logger for a specific module or component.
        This allows for more granular logging control.
    """
    return logging.getLogger(f"{settings.app_name}.{name}")

logger = setup_logging()  # Initialize logging when the module is imported






