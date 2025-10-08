"""
Logger Setup Module.

Provides a colored console logger and file logging for the image compressor.
"""

import logging


class LevelColorFormatter(logging.Formatter):
    """
    Colored console log formatter.

    Colors:
        DEBUG    -> Gray
        INFO     -> Green
        WARNING  -> Yellow
        ERROR    -> Red
        CRITICAL -> Magenta
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
        Format log records with colored level names.

        Args:
            record (logging.LogRecord): Log record.

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
    Setup a logger with colored console output and file logging.

    Args:
        name (str, optional): Logger name. Defaults to "ImageCompressor".
        log_file (str, optional): File path for persistent logs. Defaults to "compressor.log".

    Returns:
        logging.Logger: Configured logger instance.

    Notes:
        - Console output is colored.
        - File logs are plain text.
        - Prevents duplicate handlers if logger is already configured.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(LevelColorFormatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        ))

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
