import importlib
from pathlib import Path

from .logger import colorize, get_logger

logger = get_logger("core")

# 框架自身 / 库，不当插件扫。status 是插件，draw 只是绘图库。
_SKIP = {"core", "utils", "llm", "draw", "__pycache__"}
_PURPLE = "\033[35m"


def _plugin_label(name: str) -> str:
    return f'"{colorize(name, _PURPLE)}"'


def load_plugins() -> list[str]:
    """加载 src/ 下一层带 __init__.py 的包（chat 等）。utils.handler 先加载，保证指令分发挂上。"""
    importlib.import_module("src.utils.handler")
    src_dir = Path(__file__).resolve().parents[1]
    loaded: list[str] = []
    logger.info("Loading plugin modules.......")
    for path in sorted(src_dir.iterdir()):
        if not path.is_dir() or path.name in _SKIP:
            continue
        if not (path / "__init__.py").exists():
            continue
        name = f"src.{path.name}"
        try:
            importlib.import_module(name)
        except Exception:
            logger.exception("Plugin Failed %s", _plugin_label(path.name))
            continue
        loaded.append(name)
        logger.info("Plugin Succeeded %s", _plugin_label(path.name))
    return loaded
