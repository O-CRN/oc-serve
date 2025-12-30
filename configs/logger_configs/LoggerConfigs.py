"""Logger Configuration Settings and Formatters for OC-Serve."""
import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggerConfigs(BaseSettings):
    """OC-Serve Logger Configuration Settings."""
    model_config = SettingsConfigDict(env_prefix="OC_LOG_", case_sensitive=False)

    file: str = Field(default="/home/ray/logs/llm_serve.log")
    console_level: str = Field(default="INFO")
    file_level: str = Field(default="DEBUG")
    max_bytes: int = Field(default=20 * 1024 * 1024)
    backup_count: int = Field(default=7)
    base_level: str = Field(default="DEBUG")
    propagate: bool = Field(default=False)


class ColorFormatter(logging.Formatter):
    """Logging Formatter with color support for console output."""
    COLORS = {
        logging.DEBUG: "\x1b[38;5;244m",
        logging.INFO: "\x1b[38;5;34m",
        logging.WARNING: "\x1b[38;5;214m",
        logging.ERROR: "\x1b[38;5;196m",
        logging.CRITICAL: "\x1b[1;31m",
    }
    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__(
            fmt="[%(asctime)s] %(levelname)s | %(name)s | pid=%(process)d tid=%(threadName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


class PlainFormatter(logging.Formatter):
    """Logging Formatter without color for file output."""
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(name)s | pid=%(process)d tid=%(threadName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
