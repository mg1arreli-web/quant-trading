"""
Centralized logging configuration for the Quant Trading system.
Call setup_logging() once from each entry point (daemon.py, main.py).
"""
import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a clean, consistent format."""
    root_logger = logging.getLogger()

    # Avoid duplicate handlers if called multiple times
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(handler)
