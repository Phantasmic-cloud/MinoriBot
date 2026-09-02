from __future__ import annotations
import asyncio
import itertools
import json
from datetime import datetime
from typing import Any, Awaitable, Callable, Iterable

from .api import APIMixin
from .events import Event, MessageEvent
from .exceptions import ActionFailed, ApiTimeout, MinoriError
from .logger import get_logger
from .message import MessageLike, Segment

logger = get_logger("core")

_bots: dict[str, Bot] = {}
_echo_seq = itertools.count(1)
_connect_hooks: list[Callable[[Bot], Any]] = []
_disconnect_hooks: list[Callable[[Bot], Any]] = []
BotHook = Callable[["Bot"], Awaitable[Any] | Any]


class Bot(APIMixin):
    """一条已连接的 OneBot 会话。插件拿到的就是这个对象。"""

    def __init__(self, websocket: Any, self_id: int = 0, api_timeout: float = 30.0) -> None:
        self.ws = websocket
        self.self_id = int(self_id)
        self.api_timeout = api_timeout
        self._pending: dict[str, asyncio.Future] = {}
        self._send_lock = asyncio.Lock()
        self._closed = asyncio.Event()
        self.connected_at: datetime | None = None

    @property
    def connected(self) -> bool:
        return not self._closed.is_set()

    async def call_api(self, action: str, **params: Any) -> Any:
        if self._closed.is_set():
            raise MinoriError(f"Bot {self.self_id} 已断开")
        echo = f"minori_{next(_echo_seq)}"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[echo] = fut
        payload = {"action": action, "params": params, "echo": echo}
        try:
            await self._send_json(payload)
            raw = await asyncio.wait_for(asyncio.shield(fut), timeout=self.api_timeout)
        except asyncio.TimeoutError:
            raise ApiTimeout(action, self.api_timeout) from None
        except Exception:
            raise
        finally:
            self._pending.pop(echo, None)
            if not fut.done():
                fut.cancel()

        try:
            retcode = int(raw.get("retcode") or 0)
        except (TypeError, ValueError):
            retcode = -1
        status = str(raw.get("status") or "")
        if retcode != 0 and status != "ok":
            raise ActionFailed(
                action,
                retcode=retcode,
                message=str(raw.get("message") or ""),
                wording=str(raw.get("wording") or ""),
                data=raw.get("data"),
            )
        return raw.get("data")

    async def send(
        self,
        event: Event,
        message: MessageLike,
        at_sender: bool = False,
        reply: bool = False,
        **extra: Any,
    ) -> Any:
        """按事件类型回消息。群聊走 send_group_msg，私聊走 send_private_msg。"""
        if not isinstance(event, MessageEvent):
            raise MinoriError(f"无法向 {event.name} 直接发消息，请用 send_group_msg / send_private_msg")
        segs: list[Any] = []
        if reply and event.message_id:
            segs.append(Segment.reply(event.message_id))
        if at_sender and event.is_group and event.user_id:
            segs.append(Segment.at(event.user_id))
            segs.append(Segment.text(" "))
        segs.append(message)
        if event.is_group:
            return await self.send_group_msg(event.group_id, segs, **extra)
        return await self.send_private_msg(event.user_id, segs, **extra)

    def deliver(self, raw: dict[str, Any]) -> bool:
        echo = raw.get("echo")
        if echo is None or echo == "":
            return False
        fut = self._pending.get(str(echo))
        if fut is None:
            return False
        if not fut.done():
            fut.set_result(raw)
        return True

    async def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(MinoriError(f"Bot {self.self_id} 已断开"))
        self._pending.clear()
        try:
            await self.ws.close()
        except Exception:
            pass

    async def _send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        async with self._send_lock:
            await self.ws.send(data)

    def __repr__(self) -> str:
        return f"<Bot self_id={self.self_id} connected={self.connected}>"


def on_bot_connect(func: BotHook) -> BotHook:
    """注册账号连上时要跑的钩子。"""
    _connect_hooks.append(func)
    return func


def on_bot_disconnect(func: BotHook) -> BotHook:
    """注册账号掉线时要跑的钩子。"""
    _disconnect_hooks.append(func)
    return func


def _fire_hooks(hooks: list[Callable[[Bot], Any]], bot: Bot, kind: str) -> None:
    for func in list(hooks):
        try:
            result = func(bot)
            if asyncio.iscoroutine(result):
                task = asyncio.create_task(result)
                task.add_done_callback(
                    lambda t, name=getattr(func, "__name__", repr(func)): (
                        logger.exception("bot %s 钩子 %s 失败", kind, name, exc_info=t.exception())
                        if not t.cancelled() and t.exception() is not None
                        else None
                    )
                )
        except Exception:
            logger.exception("bot %s 钩子执行失败", kind)


def register_bot(bot: Bot) -> None:
    if not bot.self_id:
        return
    key = str(bot.self_id)
    old = _bots.get(key)
    if old is bot:
        return
    if old is not None:
        logger.warning("self_id=%s 重复连接，替换旧会话", key)
        _bots.pop(key, None)
        _fire_hooks(_disconnect_hooks, old, "disconnect")
        asyncio.create_task(old.close())
    bot.connected_at = datetime.now()
    _bots[key] = bot
    _fire_hooks(_connect_hooks, bot, "connect")


def unregister_bot(bot: Bot) -> None:
    key = str(bot.self_id)
    if _bots.get(key) is bot:
        _bots.pop(key, None)
        _fire_hooks(_disconnect_hooks, bot, "disconnect")


def get_bots() -> dict[str, Bot]:
    return dict(_bots)


def get_bot(self_id: int | str | None = None) -> Bot:
    if self_id is None:
        if not _bots:
            raise MinoriError("当前没有已连接的 Bot")
        if len(_bots) > 1:
            raise MinoriError(f"存在多个 Bot，请指定 self_id: {', '.join(_bots)}")
        return next(iter(_bots.values()))
    bot = _bots.get(str(self_id))
    if bot is None:
        raise MinoriError(f"Bot {self_id} 未连接")
    return bot


def iter_bots() -> Iterable[Bot]:
    return list(_bots.values())