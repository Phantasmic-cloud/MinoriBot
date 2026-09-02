import logging
import os
import sys

LOGGER_ROOT = "minori"


class BotLogger(logging.Logger):
    def print_exc(self, msg: str | None = None) -> None:
        self.error(msg or "Exception", exc_info=True)


logging.setLoggerClass(BotLogger)

_RESET = "\033[0m"
_TIME = "\033[37m"
_TAG = "\033[1;36m"
_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}


def _want_color(enabled: bool) -> bool:
    if not enabled:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        tag = record.name.split(".")[-1] if record.name else "core"
        time_s = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname
        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        if not self.use_color:
            return f"{time_s} [{level}] [{tag}] {msg}"
        level_c = _LEVEL_COLORS.get(level, "")
        return (
            f"{_TIME}{time_s}{_RESET} "
            f"{level_c}[{level}]{_RESET} "
            f"{_TAG}[{tag}]{_RESET} "
            f"{msg}"
        )


def get_logger(name: str = "core") -> BotLogger:
    return logging.getLogger(f"{LOGGER_ROOT}.{name}")  # type: ignore[return-value]


def colorize(text: str, ansi: str) -> str:
    root = logging.getLogger(LOGGER_ROOT)
    for handler in root.handlers:
        formatter = handler.formatter
        if isinstance(formatter, ColorFormatter) and formatter.use_color:
            return f"{ansi}{text}{_RESET}"
    return text


def setup_logging(level: str = "INFO", color: bool = True) -> None:
    root = logging.getLogger(LOGGER_ROOT)
    use_color = _want_color(color)
    formatter = ColorFormatter(use_color=use_color)
    log_level = getattr(logging, str(level).upper(), logging.INFO)
    if root.handlers:
        root.setLevel(log_level)
        for handler in root.handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    root.addHandler(handler)
    root.setLevel(log_level)
    root.propagate = False
