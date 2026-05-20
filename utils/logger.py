"""
Logging setup — console handler, level from LOG_LEVEL in settings.

Functions:
    setup_logger — return a named logger with consistent format
"""

import logging
import sys

from utils.config import get_settings


def setup_logger(name: str = "health_meal_planner") -> logging.Logger:
    """
    Configure and return a module-level logger.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        Configured logger instance.
    """
    settings = get_settings()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
