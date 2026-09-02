from __future__ import annotations
import json
import os
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

from .config import ROOT_DIR
from .logger import get_logger

logger = get_logger("core")
_dbs: dict[str, FileDB] = {}


def _resolve_db_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p.resolve()


def _json_default(obj: Any) -> Any:
    # Enum 按 value 写出，方便 JSON 落盘
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class FileDB:
    """按模块落盘的 JSON。set/delete 后立刻原子写入。"""

    def __init__(self, path: str | Path) -> None:
        self.path = _resolve_db_path(path)
        self.data: dict[str, Any] = {}
        self.loaded = False

    def _ensure_load(self) -> None:
        if self.loaded:
            return
        try:
            with self.path.open(encoding="utf-8") as f:
                data = json.load(f)
            self.data = data if isinstance(data, dict) else {}
            logger.debug("加载数据库 %s", self.path)
        except FileNotFoundError:
            self.data = {}
        except Exception:
            logger.exception("加载数据库 %s 失败，使用空数据", self.path)
            self.data = {}
        self.loaded = True

    def _split(self, key: str) -> list[str]:
        if not isinstance(key, str):
            raise TypeError(f"key 必须是字符串，当前类型: {type(key)}")
        key = key.replace(r"\.", "\x00")
        return [part.replace("\x00", ".") for part in key.split(".")]

    def _walk(self, key: str, create: bool = False) -> tuple[dict[str, Any] | None, str | None]:
        self._ensure_load()
        parts = self._split(key)
        last_key = parts.pop()
        cur: Any = self.data
        for part in parts:
            if not isinstance(cur, dict):
                return None, None
            if part not in cur or not isinstance(cur[part], dict):
                if not create:
                    return None, None
                cur[part] = {}
            cur = cur[part]
        if not isinstance(cur, dict):
            return None, None
        return cur, last_key

    def keys(self) -> list[str]:
        self._ensure_load()
        return list(self.data.keys())

    def get(self, key: str, default: Any = None) -> Any:
        d, k = self._walk(key)
        if d is None or k is None:
            return default
        return d.get(k, default)

    def get_copy(self, key: str, default: Any = None) -> Any:
        return deepcopy(self.get(key, default))

    def set(self, key: str, value: Any) -> None:
        d, k = self._walk(key, create=True)
        if d is None or k is None:
            raise KeyError(f"无法写入 {key}")
        d[k] = value
        self.save()

    def delete(self, key: str) -> None:
        d, k = self._walk(key)
        if d is None or k is None or k not in d:
            return
        del d[k]
        self.save()

    def save(self) -> None:
        self._ensure_load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        payload = json.dumps(self.data, ensure_ascii=False, indent=2, default=_json_default)
        with tmp.open("w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        try:
            tmp.unlink(missing_ok=True)
        except TypeError:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def get_file_db(path: str) -> FileDB:
    key = str(_resolve_db_path(path))
    db = _dbs.get(key)
    if db is None:
        db = FileDB(key)
        _dbs[key] = db
    return db


core_db = get_file_db("data/core/db.json")


def get_cached_group_name(group_id: int) -> str:
    name = core_db.get(f"groups.{int(group_id)}.name", "")
    return str(name) if name else ""


def cache_group_name(group_id: int, name: str) -> None:
    name = (name or "").strip()
    if not group_id or not name:
        return
    old = get_cached_group_name(group_id)
    if old == name:
        return
    core_db.set(f"groups.{int(group_id)}.name", name)