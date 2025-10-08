"""
Logger Setup Module

Provides a colored console logger and file logging for the image compressor.
"""

import logging

class LevelColorFormatter(logging.Formatter):
    """
    Formatter that colors only the levelname in console logs.

    Colors:
        DEBUG    -> Gray
        INFO     -> Green
        WARNING  -> Yellow
        ERROR    -> Red
        CRITICAL -> Magenta

    Example:
        12:00:01 | INFO    | This is an info message
    """

    COLORS = {
        'DEBUG': '\033[90m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[95m'
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with colored levelname.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: Formatted log string with colored level name.
        """
        original_levelname = record.levelname
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        formatted = super().format(record)
        record.levelname = original_levelname
        return formatted

def setup_logger(name: str = "ImageCompressor", log_file: str = "compressor.log") -> logging.Logger:
    """
    Create a logger with colored console output and file logging.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file for persistent logging.

    Returns:
        logging.Logger: Configured logger instance.

    Notes:
        - If the logger already has handlers, new handlers are not added to prevent duplicate logs.
        - Console logs have colored level names; file logs are plain text.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(LevelColorFormatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        ))

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
