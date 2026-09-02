from src.utils import *

from .collectors import collect_all, drop_counters, ensure_counters, note_recv, note_send
from .render import render_status

logger = get_logger("status")

_SEND_APIS = {"send_private_msg", "send_group_msg", "send_msg", "send_group_forward_msg", "send_private_forward_msg"}


# ======================= 逻辑处理 ======================= #


@on_bot_connect
async def _on_connect(bot: Bot) -> None:
    """账号连上后开始记收发条数。"""
    ensure_counters(bot.self_id)
    if getattr(bot, "_status_send_hooked", False):
        return
    orig = bot.call_api

    async def wrapped(action: str, **params):
        ret = await orig(action, **params)
        if action in _SEND_APIS:
            note_send(bot.self_id)
        return ret

    bot.call_api = wrapped  # type: ignore[method-assign]
    bot._status_send_hooked = True  # type: ignore[attr-defined]


@on_bot_disconnect
async def _on_disconnect(bot: Bot) -> None:
    """账号掉线后丢掉它的计数器。"""
    drop_counters(bot.self_id)


@on_message(priority=-100)
async def _count_recv(bot: Bot, event: MessageEvent) -> None:
    """统计收到的消息条数。"""
    if event.is_sent:
        return
    note_recv(bot.self_id)


async def get_status_image_cq(bot: Bot | None = None) -> str:
    """采集状态并画成图，返回 CQ 码。"""
    data = await collect_all(bot)
    img = await render_status(data)
    logger.info("状态图尺寸 %sx%s", img.size[0], img.size[1])
    return await get_image_cq(img, low_quality=True)


__all__ = ["get_status_image_cq"]