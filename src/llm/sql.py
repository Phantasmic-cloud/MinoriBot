# LLM 调用记录：先写内存缓存（评估用），再定时批量落盘 sqlite
import asyncio
import time
from collections import deque
from typing import Callable

import aiosqlite

from src.utils import *

logger = get_logger("llm")

DB_PATH = "data/llm/llm.sqlite"

_conn: aiosqlite.Connection | None = None
_flush_lock = asyncio.Lock()

# ===== 内存缓存：评估只看这里 =====
# key = (provider, model, kind)，value = deque[dict]
# 每条记录：{time, ok, latency_ms, error, db_id(落盘后回填，未落盘为 None)}
_CACHE_MAX = 30          # 缓存条数上限，运行时按 eval_limit*3 调整
_recent_cache: dict[tuple, deque] = {}
_disk_cached: set[tuple] = set()     # 该模型的磁盘历史已回填进内存
_pending_insert: list[dict] = []     # 待落盘队列（含落盘所需字段）

# 每次记录后触发的异步回调（事件驱动推送用）
_on_record_hooks: list[Callable] = []


def register_on_record(hook: Callable) -> None:
    """注册「每次调用记录后」的异步回调。"""
    if hook not in _on_record_hooks:
        _on_record_hooks.append(hook)


def _cache_key(provider, model, kind) -> tuple:
    return (str(provider), str(model), str(kind))


def set_cache_max(n: int) -> None:
    """由调用方（评估模块）按 eval_limit*3 设置缓存上限。"""
    global _CACHE_MAX
    if int(n) > 0:
        _CACHE_MAX = int(n)


async def _load_recent_from_db(key: tuple, limit: int) -> list[dict]:
    """从 sqlite 读该模型最近 limit 条（时间正序）。"""
    provider, model, kind = key
    conn = await _get_conn()
    cursor = await conn.execute(
        "SELECT id, time, ok, latency_ms, error FROM llm_calls"
        " WHERE provider = ? AND model = ? AND kind = ? ORDER BY id DESC LIMIT ?",
        (provider, model, kind, int(limit)),
    )
    rows = await cursor.fetchall()
    await cursor.close()
    out = [
        {"db_id": int(rid), "time": float(ts), "ok": int(ok),
         "latency_ms": int(lat or 0), "error": (str(err) if err else None)}
        for rid, ts, ok, lat, err in rows
    ]
    out.reverse()
    return out


async def _ensure_disk_cached(key: tuple) -> None:
    """首次用到该模型时，把磁盘历史回填进内存（不足条数也算成功）。"""
    if key in _disk_cached:
        return
    try:
        disk = await _load_recent_from_db(key, _CACHE_MAX)
    except Exception as e:
        logger.warning("回填 %s 历史失败: %s", key, get_exc_desc(e))
        return
    dq = _recent_cache.get(key)
    if dq is None:
        dq = deque(maxlen=_CACHE_MAX)
        _recent_cache[key] = dq
    # 合并：磁盘记录 + 内存中尚未落盘的记录（按 db_id 去重，未落盘的排最后）
    merged: list[dict] = []
    seen: set[int] = set()
    for rec in disk:
        merged.append(rec)
        seen.add(rec["db_id"])
    for rec in list(dq):
        rid = rec.get("db_id")
        if rid is None or rid not in seen:
            merged.append(rec)
            if rid is not None:
                seen.add(rid)
    merged.sort(key=lambda r: (r["time"], r.get("db_id") or 0))
    _recent_cache[key] = deque(merged[-_CACHE_MAX:], maxlen=_CACHE_MAX)
    _disk_cached.add(key)


async def get_recent_calls(provider, model, kind, limit: int) -> list[dict]:
    """取最近 limit 次调用（时间正序）。仅读内存。"""
    key = _cache_key(provider, model, kind)
    await _ensure_disk_cached(key)
    dq = _recent_cache.get(key) or deque()
    return list(dq)[-int(limit):]


async def flush_pending() -> None:
    """把待落盘队列批量写进 sqlite，成功后回填 db_id。"""
    if not _pending_insert:
        return
    async with _flush_lock:
        if not _pending_insert:
            return
        batch = _pending_insert[:]
        _pending_insert.clear()
        done: list[tuple[dict, int]] = []
        try:
            conn = await _get_conn()
            for rec in batch:
                cursor = await conn.execute(
                    "INSERT INTO llm_calls (time, provider, model, kind, ok, latency_ms,"
                    " prompt_tokens, completion_tokens, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rec["time"], rec["provider"], rec["model"], rec["kind"],
                        1 if rec["ok"] else 0, rec["latency_ms"],
                        rec["prompt_tokens"], rec["completion_tokens"], rec["error"],
                    ),
                )
                rid = int(getattr(cursor, "lastrowid", 0) or 0)
                if rid:
                    done.append((rec, rid))
            await conn.commit()
        except Exception as e:
            logger.warning("批量落盘 %d 条记录失败: %s", len(batch), get_exc_desc(e))
            # 整批回滚、db_id 复位，退回队列下次重试（避免半写状态产生重复记录）
            try:
                await (await _get_conn()).rollback()
            except Exception:
                pass
            _pending_insert[:0] = batch
            return
        # commit 成功后才回填 db_id
        for rec, rid in done:
            rec["db_id"] = rid


@repeat_with_interval(
    Config("llm.llm").item("model_api_status.notify.flush_interval"),
    "LLM统计落盘",
    logger,
)
async def _flush_loop() -> None:
    """定时把待落盘队列批量写进 sqlite。"""
    await flush_pending()


async def _get_conn() -> aiosqlite.Connection:
    global _conn
    if _conn is None:
        create_parent_folder(DB_PATH)
        _conn = await aiosqlite.connect(DB_PATH)
        await _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time REAL NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                kind TEXT NOT NULL,
                ok INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                error TEXT
            )
            """
        )
        await _conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_time ON llm_calls(time)")
        await _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_model_time"
            " ON llm_calls(provider, model, kind, time)"
        )
        await _conn.commit()
        logger.info("连接LLM统计数据库 %s 成功", DB_PATH)
    return _conn


async def record_call(provider, model, kind: str, ok: bool, latency_ms: int = 0,
                      prompt_tokens: int = 0, completion_tokens: int = 0,
                      error: str = None) -> None:
    """记录一次 LLM / embedding / TTS 调用：只写内存，秒回；落盘交给定时任务。

    任何失败只 warning，不影响主流程。评估回调在内存写入后立即触发。
    """
    try:
        key = _cache_key(provider, model, kind)
        rec = {
            "time": time.time(),
            "ok": 1 if ok else 0,
            "latency_ms": int(latency_ms or 0),
            "error": None if ok else (truncate(str(error or ""), 200) or None),
            "db_id": None,
            # 落盘用字段
            "provider": key[0], "model": key[1], "kind": key[2],
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
        }
        dq = _recent_cache.get(key)
        if dq is None:
            dq = deque(maxlen=_CACHE_MAX)
            _recent_cache[key] = dq
        dq.append(rec)
        _pending_insert.append(rec)
    except Exception as e:
        logger.warning("记录LLM调用失败: %s", get_exc_desc(e))
        return
    _fire_record_hooks(provider, model, kind)


async def _run_record_hook(hook: Callable, provider, model, kind) -> None:
    try:
        # 先补磁盘历史（首次），再评估；这样重启后第一次评估就能看到完整窗口
        await _ensure_disk_cached(_cache_key(provider, model, kind))
        await hook(provider, model, kind)
    except Exception as e:
        logger.warning("调用记录回调失败: %s", get_exc_desc(e))


def _fire_record_hooks(provider, model, kind) -> None:
    for hook in list(_on_record_hooks):
        try:
            asyncio.get_running_loop().create_task(_run_record_hook(hook, provider, model, kind))
        except RuntimeError:
            pass