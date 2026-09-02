from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
_RELOAD_INTERVAL = 3.0


def _norm_ws_path(path: str) -> str:
    path = path.split("?", 1)[0].strip() or "/"
    if not path.startswith("/"):
        path = "/" + path
    if path != "/":
        path = path.rstrip("/")
    return path


def config_path(name: str, config_dir: str | Path | None = None) -> Path:
    """core -> config/core.yaml，llm.openai -> config/llm/openai.yaml"""
    base = Path(config_dir) if config_dir else CONFIG_DIR
    return base / (str(name).replace(".", "/") + ".yaml")


class ConfigItem:
    def __init__(self, config: Config, key: str | tuple[str, ...]) -> None:
        self.config = config
        self.key = key

    def get(self, default: Any = None, raise_exc: bool | None = None) -> Any:
        if raise_exc is None:
            raise_exc = default is None
        return self.config.get(self.key, default, raise_exc=raise_exc)


def get_cfg_or_value(obj: Any, default: Any = None, raise_exc: bool | None = None) -> Any:
    if isinstance(obj, ConfigItem):
        return obj.get(default, raise_exc)
    return obj


def parse_cfg_num(x: Any) -> int | float:
    if isinstance(x, (int, float)):
        return x
    try:
        return eval(str(x), {"__builtins__": None}, {})
    except Exception as e:
        raise ValueError(f"无法解析配置数字 '{x}': {e}") from e


class Config:
    """按名字读 config/ 下的 yaml。改文件后最多 3 秒生效。"""

    _cache: dict[str, tuple[int, dict[str, Any]]] = {}

    def __init__(self, name: str, config_dir: str | Path | None = None) -> None:
        self.name = name
        self.dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.path = config_path(name, self.dir)
        self._last_check = 0.0
        self._load(force=True)

    def _load(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_check < _RELOAD_INTERVAL:
            return
        self._last_check = now
        key = str(self.path)
        if not self.path.exists():
            self._cache[key] = (0, {})
            return
        mtime = int(self.path.stat().st_mtime)
        cached = self._cache.get(key)
        if not force and cached and cached[0] == mtime:
            return
        with self.path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{self.path} 必须是一个映射")
        self._cache[key] = (mtime, data)

    def get_all(self) -> dict[str, Any]:
        self._load()
        return dict(self._cache.get(str(self.path), (0, {}))[1])

    def get(
        self,
        key: str | tuple[str, ...] | None = None,
        default: Any = None,
        raise_exc: bool | None = None,
        *,
        required: bool = False,
    ) -> Any:
        data: Any = self.get_all()
        if key is None:
            return data
        keys = key.split(".") if isinstance(key, str) else list(key)
        cur = data
        for part in keys:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                if required or (raise_exc if raise_exc is not None else False):
                    raise KeyError(f"配置 {self.name} 中不存在 {key}")
                return default
        return cur

    def item(self, key: str | tuple[str, ...]) -> ConfigItem:
        return ConfigItem(self, key)

    def mtime(self) -> int:
        self._load()
        return int(self._cache.get(str(self.path), (0, {}))[0])


@dataclass
class CoreConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    path: str = "/onebot/v11/ws"
    access_token: str = ""
    api_timeout: float = 30.0
    log_level: str = "INFO"
    log_color: bool = True
    max_message_size: int = 16 * 1024 * 1024

    @classmethod
    def load(cls, config_dir: str | Path | None = None) -> CoreConfig:
        file = Config("core", config_dir)
        if not file.path.exists():
            raise FileNotFoundError(f"找不到配置文件: {file.path}")
        data = file.get_all()
        server = data.get("server") if isinstance(data.get("server"), dict) else data
        defaults = cls()
        return cls(
            host=str(server.get("host", defaults.host)),
            port=int(server.get("port", defaults.port)),
            path=_norm_ws_path(str(server.get("path", defaults.path))),
            access_token=str(server.get("access_token", defaults.access_token) or ""),
            api_timeout=float(server.get("api_timeout", defaults.api_timeout)),
            log_level=str(server.get("log_level", defaults.log_level)),
            log_color=bool(server.get("log_color", defaults.log_color)),
            max_message_size=int(server.get("max_message_size", defaults.max_message_size)),
        )
