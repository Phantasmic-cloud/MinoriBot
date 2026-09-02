from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from .events import Event, MessageEvent, MetaEvent, NoticeEvent, RequestEvent
from .exceptions import StopPropagation
from .logger import get_logger

if TYPE_CHECKING:
    from .bot import Bot

Handler = Callable[["Bot", Event], Awaitable[Any] | Any]
logger = get_logger("core")


@dataclass
class HandlerEntry:
    func: Handler
    event_type: type[Event]
    priority: int = 10
    block: bool = False
    name: str = ""


class EventBus:
    def __init__(self) -> None:
        self._handlers: list[HandlerEntry] = []

    def subscribe(
        self,
        func: Handler,
        event_type: type[Event] = Event,
        priority: int = 10,
        block: bool = False,
    ) -> Handler:
        self._handlers.append(
            HandlerEntry(
                func=func,
                event_type=event_type,
                priority=priority,
                block=block,
                name=getattr(func, "__name__", repr(func)),
            )
        )
        self._handlers.sort(key=lambda h: h.priority)
        return func

    def on(
        self,
        event_type: type[Event] = Event,
        priority: int = 10,
        block: bool = False,
    ) -> Callable[[Handler], Handler]:
        def decorator(func: Handler) -> Handler:
            return self.subscribe(func, event_type, priority, block)

        return decorator

    async def emit(self, bot: Bot, event: Event) -> None:
        for entry in list(self._handlers):
            if not isinstance(event, entry.event_type):
                continue
            try:
                result = entry.func(bot, event)
                if asyncio.iscoroutine(result):
                    await result
            except StopPropagation:
                logger.debug("handler %s 中止后续处理", entry.name)
                return
            except Exception:
                logger.exception("handler %s 处理 %s 出错", entry.name, event.name)
                continue
            if entry.block:
                return


bus = EventBus()


def on_event(
    event_type: type[Event] = Event,
    priority: int = 10,
    block: bool = False,
) -> Callable[[Handler], Handler]:
    return bus.on(event_type, priority, block)


def on_message(priority: int = 10, block: bool = False) -> Callable[[Handler], Handler]:
    return bus.on(MessageEvent, priority, block)


def on_notice(priority: int = 10, block: bool = False) -> Callable[[Handler], Handler]:
    return bus.on(NoticeEvent, priority, block)


def on_request(priority: int = 10, block: bool = False) -> Callable[[Handler], Handler]:
    return bus.on(RequestEvent, priority, block)


def on_meta(priority: int = 10, block: bool = False) -> Callable[[Handler], Handler]:
    return bus.on(MetaEvent, priority, block)