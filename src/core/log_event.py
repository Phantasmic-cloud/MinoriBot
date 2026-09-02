from .bot import Bot
from .events import Event, MessageEvent, NoticeEvent, RequestEvent
from .logger import get_logger
from .message import Message
from .store import cache_group_name, get_cached_group_name

logger = get_logger("core")
_seen_msg_ids: set[int] = set()
_SEEN_LIMIT = 4096


def remember_logged_msg(msg_id: int) -> bool:
    """记录已打过日志的 message_id。返回 False 表示已经记过，应跳过。"""
    if not msg_id:
        return True
    if msg_id in _seen_msg_ids:
        return False
    _seen_msg_ids.add(msg_id)
    if len(_seen_msg_ids) > _SEEN_LIMIT:
        _seen_msg_ids.clear()
    return True


def summarize_message(message: Message, limit: int = 160) -> str:
    parts: list[str] = []
    for seg in message:
        t = seg.type
        data = seg.data or {}
        if t == "text":
            parts.append(str(data.get("text") or ""))
        elif t == "at":
            qq = data.get("qq")
            parts.append("[@全体成员]" if str(qq) == "all" else f"[@{qq}]")
        elif t == "image":
            parts.append("[表情]" if str(data.get("sub_type") or "0") not in ("", "0") else "[图片]")
        elif t == "face":
            parts.append("[表情]")
        elif t == "mface":
            parts.append("[表情]")
        elif t == "record":
            parts.append("[语音]")
        elif t == "video":
            parts.append("[视频]")
        elif t == "file":
            name = data.get("name") or data.get("file") or ""
            parts.append(f"[文件:{name}]" if name else "[文件]")
        elif t == "reply":
            parts.append(f"[回复:{data.get('id')}]")
        elif t == "forward":
            parts.append("[转发]")
        elif t == "json":
            parts.append("[卡片]")
        elif t == "xml":
            parts.append("[XML]")
        elif t == "share":
            parts.append(f"[分享:{data.get('title') or data.get('url') or ''}]")
        elif t == "poke":
            parts.append("[戳一戳]")
        else:
            parts.append(f"[{t}]")
    text = "".join(parts).replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text or "[空消息]"


def _group_label(event: Event) -> str:
    group_id = int(getattr(event, "group_id", 0) or 0)
    if not group_id:
        return ""
    name = str(event.get("group_name") or "") or get_cached_group_name(group_id)
    if name:
        cache_group_name(group_id, name)
        return f"{name}({group_id})"
    return str(group_id)


def _user_label(event: MessageEvent) -> str:
    name = event.sender.display_name
    return f"{name}({event.user_id})" if name else str(event.user_id)


async def _resolve_group_name(bot: Bot, event: MessageEvent) -> str:
    group_id = event.group_id
    cached = _group_label(event)
    if cached and cached != str(group_id):
        return cached
    try:
        info = await bot.get_group_info(group_id)
        name = ""
        if isinstance(info, dict):
            name = str(info.get("group_name") or "")
        if name:
            cache_group_name(group_id, name)
            return f"{name}({group_id})"
    except Exception:
        logger.debug("获取群名失败 group_id=%s", group_id, exc_info=True)
    return str(group_id)


def _log_notice(event: NoticeEvent) -> None:
    ntype = event.notice_type
    subtype = event.sub_type
    group = _group_label(event)
    if ntype == "group_recall":
        logger.info("群 %s 的用户 %s 撤回了用户 %s 的消息 %s", group, event.operator_id, event.user_id, event.message_id)
    elif ntype == "friend_recall":
        logger.info("用户 %s 撤回了私聊消息 %s", event.user_id, event.message_id)
    elif ntype == "group_increase":
        logger.info("群 %s 新成员 %s 操作者 %s", group, event.user_id, event.operator_id)
    elif ntype == "group_decrease":
        logger.info("群 %s 成员 %s 离开/被踢 操作者 %s", group, event.user_id, event.operator_id)
    elif ntype == "group_ban":
        logger.info("群 %s 用户 %s 被 %s 禁言 %ss", group, event.user_id, event.operator_id, event.duration)
    elif ntype == "notify" and subtype == "poke":
        logger.info("群 %s 的用户 %s 戳了用户 %s", group or "私聊", event.user_id, event.target_id)
    elif ntype == "group_msg_emoji_like":
        likes = event.likes or []
        if not likes:
            logger.info("群 %s 的用户 %s 给消息 %s 回应了表情", group, event.user_id, event.message_id)
        else:
            for like in likes:
                if not isinstance(like, dict):
                    continue
                logger.info(
                    "群 %s 的用户 %s 给消息 %s 回应了 %s 个emoji %s",
                    group,
                    event.user_id,
                    event.message_id,
                    like.get("count"),
                    like.get("emoji_id"),
                )
    elif ntype == "notify" and subtype == "honor":
        logger.info("群 %s 用户 %s 获得荣誉 %s", group, event.user_id, event.honor_type)
    elif ntype == "group_upload":
        name = event.file.get("name") if isinstance(event.file, dict) else ""
        logger.info("群 %s 用户 %s 上传文件 %s", group, event.user_id, name or event.file)
    else:
        extra = f".{subtype}" if subtype else ""
        logger.info("notice %s%s group=%s user=%s", ntype, extra, group or "-", event.user_id or "-")


def _log_request(event: RequestEvent) -> None:
    extra = f".{event.sub_type}" if event.sub_type else ""
    comment = (event.comment or "").replace("\n", " ")
    if event.request_type == "friend":
        logger.info("好友请求 user=%s comment=%s", event.user_id, comment)
    elif event.request_type == "group":
        logger.info("加群请求%s group=%s user=%s comment=%s", extra, _group_label(event), event.user_id, comment)
    else:
        logger.info("request %s%s user=%s", event.request_type, extra, event.user_id)


async def log_event(bot: Bot, event: Event) -> None:
    if isinstance(event, MessageEvent):
        if not remember_logged_msg(event.message_id):
            return
        summary = summarize_message(event.message)
        user = _user_label(event)
        if event.is_private:
            prefix = "自身私聊" if event.is_sent or event.user_id == event.self_id else "私聊"
            logger.info("[%s] %s %s: %s", event.message_id, prefix, user, summary)
            return
        group = await _resolve_group_name(bot, event)
        if event.is_sent or event.user_id == event.self_id:
            logger.info("[%s] %s 自身消息: %s", event.message_id, group, summary)
        else:
            logger.info("[%s] %s %s: %s", event.message_id, group, user, summary)
        return
    if isinstance(event, NoticeEvent):
        _log_notice(event)
        return
    if isinstance(event, RequestEvent):
        _log_request(event)