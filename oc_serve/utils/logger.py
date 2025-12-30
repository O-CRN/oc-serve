"""Logger Factory for OC-Serve."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from configs import LoggerConfigs, ColorFormatter, PlainFormatter


class OCLogger:
    """Singleton Logger Factory to create and manage loggers."""
    _instance: Optional["OCLogger"] = None
    _configured: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.configs = LoggerConfigs()
        if self._configured:
            return

        if isinstance(self.configs.console_level, str):
            self.configs.console_level = logging._nameToLevel.get(
                self.configs.console_level.upper(), logging.INFO
                )
        if isinstance(self.configs.file_level, str):
            self.configs.file_level = logging._nameToLevel.get(
                self.configs.file_level.upper(), logging.DEBUG
                )

        parent = os.path.dirname(os.path.abspath(self.configs.file))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        self._stdout_handler = logging.StreamHandler(sys.stdout)
        self._stdout_handler.setLevel(self.configs.console_level)
        self._stdout_handler.setFormatter(ColorFormatter())

        self._file_handler = RotatingFileHandler(
            self.configs.file, maxBytes=self.configs.max_bytes,
            backupCount=self.configs.backup_count, encoding="utf-8"
        )
        self._file_handler.setLevel(self.configs.file_level)
        self._file_handler.setFormatter(PlainFormatter())

        self._configured = True

    def get_logger(self, name: str = "oc-serve") -> logging.Logger:
        """Retrieve a logger by name, configuring it if necessary."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
            logger.addHandler(self._file_handler)
        if not any(isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
                   for h in logger.handlers):
            logger.addHandler(self._stdout_handler)
        return logger
