"""
Shared logging configuration.

Usage in any script:
    from config.logging_config import get_logger
    log = get_logger(__name__)
    log.info("Message here")
"""

import logging
import sys
from pathlib import Path

# Project root for log file location
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'


def get_logger(name: str, level: int = logging.INFO, log_to_file: bool = False) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default INFO)
        log_to_file: If True, also log to logs/<name>.log
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with simple format
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console)
    
    # Optional file handler
    if log_to_file:
        LOG_DIR.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(LOG_DIR / f'{name.split(".")[-1]}.log')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    
    return logger


def configure_root_logger(level: int = logging.INFO):
    """Configure the root logger for simple scripts."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
