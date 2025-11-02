# src/utils/logger.py

import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


def get_logger(
    name="RSNA",
    save_dir="outputs/logs",
    level=logging.INFO,
    when="midnight",
    backup_count=7,
):
    """
    Create a logger that writes to both console and rotating log files.

    Parameters
    ----------
    name : str
        Name of the logger (e.g. "RSNA_train")
    save_dir : str
        Directory where log files are stored
    level : logging level
        Logging level (default: INFO)
    when : str
        Interval for rotation (e.g., "midnight", "D", "H")
    backup_count : int
        How many old log files to keep

    Returns
    -------
    logger : logging.Logger
    """
    os.makedirs(save_dir, exist_ok=True)

    # log filename based on current date
    log_filename = os.path.join(save_dir, f"{name}.log")

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # avoid duplicate handlers
    if not logger.handlers:
        # File handler (rotates daily)
        file_handler = TimedRotatingFileHandler(
            log_filename, when=when, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Formatter
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"🧾 Logging started → {os.path.abspath(log_filename)}")
    return logger

