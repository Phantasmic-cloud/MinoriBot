# LLM 调用记录：token 用量与请求状态共用一张表，一次调用记一行
import asyncio
import time

import aiosqlite

from src.utils import *

logger = get_logger("llm")

DB_PATH = "data/llm/llm.sqlite"

_conn: aiosqlite.Connection | None = None
_write_lock = asyncio.Lock()


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
        await _conn.commit()
        logger.info("连接LLM统计数据库 %s 成功", DB_PATH)
    return _conn


async def record_call(provider, model, kind: str, ok: bool, latency_ms: int = 0,
                      prompt_tokens: int = 0, completion_tokens: int = 0,
                      error: str = None) -> None:
    """记录一次 LLM / embedding / TTS 调用。任何失败只 warning，不影响主流程。

    kind: chat / embedding / tts（tts 时 prompt_tokens 为字符数）
    成功时带 token 用量，失败时 token 记 0、error 带原因。
    """
    try:
        async with _write_lock:
            conn = await _get_conn()
            await conn.execute(
                "INSERT INTO llm_calls (time, provider, model, kind, ok, latency_ms,"
                " prompt_tokens, completion_tokens, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(), str(provider), str(model), str(kind),
                    1 if ok else 0, int(latency_ms or 0),
                    int(prompt_tokens or 0), int(completion_tokens or 0),
                    None if ok else (truncate(str(error or ""), 200) or None),
                ),
            )
            await conn.commit()
    except Exception as e:
        logger.warning("记录LLM调用失败: %s", get_exc_desc(e))
