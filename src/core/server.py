import asyncio
import http
import inspect
import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.exceptions import ConnectionClosed

from .bot import Bot, get_bots, register_bot, unregister_bot
from .bus import bus
from .config import CoreConfig
from .events import parse_event
from .log_event import log_event
from .logger import get_logger

logger = get_logger("core")
_bg_tasks: set[asyncio.Task] = set()
_stop_event: asyncio.Event | None = None


def request_stop() -> None:
    if _stop_event is None:
        logger.warning("服务尚未启动，无法停止")
        return
    _stop_event.set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)

    def _done(t: asyncio.Task) -> None:
        _bg_tasks.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.exception("后台事件处理失败", exc_info=exc)

    task.add_done_callback(_done)


def _headers_of(ws: Any) -> Any:
    request = getattr(ws, "request", None)
    if request is not None and getattr(request, "headers", None) is not None:
        return request.headers
    if getattr(ws, "request_headers", None) is not None:
        return ws.request_headers
    return {}


def _path_of(ws: Any, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    request = getattr(ws, "request", None)
    if request is not None and getattr(request, "path", None):
        return request.path
    if getattr(ws, "path", None):
        return ws.path
    return "/"


def _header(headers: Any, name: str) -> str:
    if headers is None:
        return ""
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value:
            return str(value)
        value = getter(name.lower())
        if value:
            return str(value)
    try:
        return str(headers[name])
    except Exception:
        return ""


def _extract_token(path: str, headers: Any) -> str:
    auth = _header(headers, "Authorization") or _header(headers, "authorization")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if auth:
        return auth.strip()
    query = parse_qs(urlparse(path).query)
    values = query.get("access_token") or []
    return values[0] if values else ""


def _parse_self_id(headers: Any) -> int:
    raw = _header(headers, "X-Self-ID") or _header(headers, "x-self-id")
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _norm_path(path: str) -> str:
    path = urlparse(path).path or "/"
    if path != "/":
        path = path.rstrip("/")
    return path


def _handshake_error(path: str, headers: Any, config: CoreConfig) -> tuple[http.HTTPStatus, str] | None:
    if _norm_path(path) != config.path:
        return http.HTTPStatus.NOT_FOUND, "not found"
    if config.access_token:
        token = _extract_token(path, headers)
        if token != config.access_token:
            return http.HTTPStatus.UNAUTHORIZED, "unauthorized"
    return None


def _make_process_request(config: CoreConfig):
    async def process_request(*args):
        if len(args) == 2 and isinstance(args[0], str):
            path, headers = args
        elif len(args) == 2:
            connection, request = args
            path = getattr(request, "path", "/")
            headers = getattr(request, "headers", {})
            err = _handshake_error(path, headers, config)
            if err is None:
                return None
            status, body = err
            respond = getattr(connection, "respond", None)
            if callable(respond):
                return respond(status, body)
            return status, [], body.encode("utf-8")
        else:
            return None
        err = _handshake_error(path, headers, config)
        if err is None:
            return None
        status, body = err
        return status, [], body.encode("utf-8")

    return process_request


async def _handle_payload(bot: Bot, data: dict[str, Any]) -> None:
    # 对不上 echo 的 API 回包，不当事件分发
    if data.get("echo") not in (None, "") and "post_type" not in data:
        logger.debug("丢弃未匹配的 API 回包 echo=%s", data.get("echo"))
        return
    event = parse_event(data)
    if event.self_id and event.self_id != bot.self_id:
        if bot.self_id:
            unregister_bot(bot)
        bot.self_id = event.self_id
        register_bot(bot)
    elif event.self_id and not bot.self_id:
        bot.self_id = event.self_id
        register_bot(bot)
    try:
        await log_event(bot, event)
    except Exception:
        logger.exception("记录事件失败 %s", event.name)
    await bus.emit(bot, event)


async def _handle_raw(bot: Bot, raw: str | bytes) -> None:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("收到非 JSON 消息，已忽略")
        return
    if not isinstance(data, dict):
        logger.warning("收到非对象 JSON，已忽略")
        return
    # echo 回包必须同步投递，不能丢进后台任务再等调度
    if bot.deliver(data):
        return
    _spawn(_handle_payload(bot, data))


async def _session(ws: Any, config: CoreConfig, path: str | None = None) -> None:
    headers = _headers_of(ws)
    req_path = _path_of(ws, path)
    err = _handshake_error(req_path, headers, config)
    if err is not None:
        status, reason = err
        logger.warning("拒绝连接 path=%s status=%s", _norm_path(req_path), status)
        try:
            await ws.close(code=1008, reason=reason)
        except Exception:
            pass
        return
    self_id = _parse_self_id(headers)
    bot = Bot(ws, self_id=self_id, api_timeout=config.api_timeout)
    if bot.self_id:
        register_bot(bot)
    logger.info("Connection to the OneBot Client")

    async def read_loop() -> None:
        async for raw in ws:
            await _handle_raw(bot, raw)

    async def ensure_login() -> None:
        nickname = ""
        try:
            info = await bot.get_login_info()
            if isinstance(info, dict):
                if info.get("user_id"):
                    uid = int(info["user_id"])
                    if uid != bot.self_id:
                        if bot.self_id:
                            unregister_bot(bot)
                        bot.self_id = uid
                        register_bot(bot)
                    elif not bot.self_id:
                        bot.self_id = uid
                        register_bot(bot)
                nickname = str(info.get("nickname") or "")
        except Exception:
            logger.warning("未能通过 get_login_info 获取账号信息，等待事件补全")
        logger.info("Bot qqid=%s |  Nickname=%s", bot.self_id or "?", nickname or "-")

    try:
        # 读循环必须先跑起来，否则 get_login_info 的 echo 回包会永远等不到
        await asyncio.gather(read_loop(), ensure_login())
    except ConnectionClosed:
        logger.info("Disconnect from the OneBot Client")
    except Exception:
        logger.exception("OneBot 连接异常")
    finally:
        unregister_bot(bot)
        await bot.close()
        if not get_bots():
            logger.info("当前无可用Bot")


async def serve(config: CoreConfig) -> None:
    global _stop_event
    _stop_event = asyncio.Event()

    async def handler(websocket, path=None):
        await _session(websocket, config, path)

    kwargs: dict[str, Any] = {
        "max_size": config.max_message_size,
        "ping_interval": 20,
        "ping_timeout": 20,
        "process_request": _make_process_request(config),
    }
    try:
        sig = inspect.signature(websockets.serve)
        if "origins" in sig.parameters:
            kwargs["origins"] = None
    except (TypeError, ValueError):
        pass

    logger.info("反向 WS 监听 ws://%s:%s%s", config.host, config.port, config.path)
    try:
        server = websockets.serve(handler, config.host, config.port, **kwargs)
    except TypeError:
        kwargs.pop("process_request", None)
        kwargs.pop("origins", None)
        server = websockets.serve(handler, config.host, config.port, **kwargs)
    async with server:
        await _stop_event.wait()