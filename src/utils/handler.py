import asyncio
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from src.core import (
    Bot,
    Config,
    MessageEvent,
    NoReplyException,
    ReplyException,
    StopPropagation,
    get_bot,
    get_cached_group_name,
    get_cfg_or_value,
    get_file_db,
    get_logger,
    on_message,
    remember_logged_msg,
    summarize_message,
)
from src.core.message import Message, MessageLike, Segment
from src.core.store import cache_group_name
from src.utils.utils import (
    create_parent_folder,
    find_by,
    get_exc_desc,
    get_image_cq,
    get_md5,
    get_readable_timedelta,
    get_str_display_length,
    truncate,
)

logger = get_logger("utils")
core_logger = get_logger("core")
global_cfg = Config("global")
utils_db = get_file_db("data/utils/db.json")
_bot_reply_msg_ids: set[int] = set()

DEFAULT_FOLD_THRESHOLD_CFG = global_cfg.item("msg_send.default_fold_threshold")
DEFAULT_FOLD_FALLBACK_METHOD_CFG = global_cfg.item("msg_send.default_fold_fallback_method")
MAX_FOLD_MSG_SEGMENT_COUNT_CFG = global_cfg.item("msg_send.max_fold_msg_segment_count")
MAX_FOLD_MSG_LEN_CFG = global_cfg.item("msg_send.max_fold_msg_len")
DEFAULT_CD_CFG = global_cfg.item("default_cd")
CD_VERBOSE_INTERVAL_CFG = global_cfg.item("cd_verbose_interval")
GROUP_MEMBER_NAME_CACHE_EXPIRE_CFG = global_cfg.item("group_member_name_cache_expire_seconds")
STRANGER_NAME_CACHE_EXPIRE_CFG = global_cfg.item("stranger_name_cache_expire_seconds")


class ExpirableCache:
    def __init__(self, default_expire_seconds) -> None:
        self.cache: dict[Any, tuple[Any, datetime]] = {}
        self.default_expire_seconds = default_expire_seconds

    def get(self, key: Any) -> Any | None:
        now = datetime.now()
        for k in [k for k, (_, exp) in self.cache.items() if exp <= now]:
            self.cache.pop(k, None)
        item = self.cache.get(key)
        if not item:
            return None
        value, expire_time = item
        if expire_time > now:
            return value
        self.cache.pop(key, None)
        return None

    def set(self, key: Any, value: Any, expire_seconds: int | None = None) -> None:
        seconds = expire_seconds
        if seconds is None:
            seconds = get_cfg_or_value(self.default_expire_seconds, 300)
        self.cache[key] = (value, datetime.now() + timedelta(seconds=float(seconds or 300)))


_group_member_name_cache = ExpirableCache(GROUP_MEMBER_NAME_CACHE_EXPIRE_CFG)
_stranger_name_cache = ExpirableCache(STRANGER_NAME_CACHE_EXPIRE_CFG)


def is_group_msg(event: MessageEvent) -> bool:
    return event.is_group


def check_self(event: MessageEvent) -> bool:
    return event.user_id == event.self_id


def check_self_reply(event: MessageEvent) -> bool:
    return int(event.message_id) in _bot_reply_msg_ids


def check_is_bot_reply_msg(msg_id: int) -> bool:
    return int(msg_id) in _bot_reply_msg_ids


def check_superuser(event: MessageEvent, superuser: Any = None) -> bool:
    users = get_cfg_or_value(superuser, None) if superuser is not None else global_cfg.get("superuser")
    if not users:
        return False
    return int(event.user_id) in [int(x) for x in users]


def extract_cq_code(msg) -> dict[str, list[dict]]:
    segs = Message.of(msg)
    ret: dict[str, list[dict]] = {}
    for seg in segs:
        ret.setdefault(seg.type, []).append(dict(seg.data))
    return ret


def extract_text(msg) -> str:
    return Message.of(msg).plain_text


def extract_image_url(msg) -> list[str]:
    return [d["url"] for d in extract_image_data(msg) if d.get("url")]


def extract_image_data(msg) -> list[dict]:
    ret: list[dict] = []
    for d in extract_cq_code(msg).get("image", []):
        item = dict(d)
        if not item.get("url") and item.get("file"):
            file = str(item["file"])
            if file.startswith("http"):
                item["url"] = file
        if item.get("url") or item.get("file"):
            ret.append(item)
    for d in extract_cq_code(msg).get("mface", []):
        url = d.get("url")
        if not url:
            continue
        ret.append(
            {
                "file": str(d.get("file") or url.rsplit("/", 1)[-1]),
                "url": url,
                "file_unique": str(d.get("file_unique") or url.rsplit("/", 1)[-1].split(".")[0]),
                "subType": 1,
                "summary": d.get("summary", "[表情]"),
            }
        )
    return ret


def extract_at_qq(msg) -> list[int]:
    ids: list[int] = []
    for d in extract_cq_code(msg).get("at", []):
        qq = d.get("qq")
        if qq in (None, "all"):
            continue
        try:
            ids.append(int(qq))
        except (TypeError, ValueError):
            continue
    return ids


async def upload_group_file(bot: Bot, group_id: int, file_path: str, name: str, folder: str = "/") -> dict:
    return await bot.call_api(
        "upload_group_file",
        group_id=int(group_id),
        file=f"file://{os.path.abspath(file_path)}",
        name=name,
        folder=folder,
    )


def command_text(event: MessageEvent) -> str:
    """去掉开头的回复 / @我 / 空白后再做指令匹配。@别人开头的不当指令。"""
    segs = list(event.message)
    i = 0
    self_id = str(event.self_id)
    while i < len(segs):
        seg = segs[i]
        if seg.type == "reply":
            i += 1
            continue
        if seg.type == "at" and str(seg.data.get("qq", "")) == self_id:
            i += 1
            continue
        if seg.type == "text" and not str(seg.data.get("text") or "").strip():
            i += 1
            continue
        break
    if i < len(segs) and segs[i].type == "at":
        return ""
    return Message(segs[i:]).plain_text.lstrip()


def get_msg(event: MessageEvent) -> list[dict]:
    return event.message.to_rich()


def get_reply_id(event: MessageEvent | None) -> int | None:
    if event is None:
        return None
    reply = event.get("reply")
    if isinstance(reply, dict) and reply.get("message_id"):
        try:
            return int(reply["message_id"])
        except (TypeError, ValueError):
            pass
    for seg in event.message:
        if seg.type == "reply":
            try:
                return int(seg.data.get("id"))
            except (TypeError, ValueError):
                return None
    return None


def get_reply_msg(event: MessageEvent):
    reply = event.get("reply")
    if not reply:
        return None
    if isinstance(reply, dict):
        return reply.get("message")
    return getattr(reply, "message", None)


async def aget_reply_msg(bot: Bot, event: MessageEvent):
    """优先用事件自带的 reply，没有则按 reply 段调 get_msg 补齐。"""
    msg = get_reply_msg(event)
    if msg is not None:
        return msg
    reply_id = get_reply_id(event)
    if not reply_id:
        return None
    try:
        msg_obj = await bot.get_msg(reply_id)
    except Exception:
        logger.print_exc(f"获取回复消息 {reply_id} 失败")
        return None
    if not isinstance(msg_obj, dict):
        return None
    event.raw["reply"] = msg_obj
    sender = msg_obj.get("sender") or {}
    if isinstance(sender, dict) and str(sender.get("user_id") or "") == str(event.self_id):
        event.to_me = True
    return msg_obj.get("message")


def _normalize_forward_msg(result: Any) -> dict:
    if not isinstance(result, dict):
        return {"messages": []}
    if "messages" in result:
        return result
    ret = {"messages": []}
    for node in result.get("message") or []:
        if not isinstance(node, dict):
            continue
        msg = dict(node.get("data") or {})
        if "message" not in msg and "content" in msg:
            msg["message"] = msg["content"]
        msg.setdefault("time", 0)
        ret["messages"].append(msg)
    return ret


async def get_forward_msg(bot: Bot, forward_id: str) -> dict:
    result = await bot.get_forward_msg(id=str(forward_id))
    result = _normalize_forward_msg(result)
    return result


async def get_image_datas_from_msg(
    bot: Bot,
    msg_or_event,
    parse_reply: bool = True,
    parse_forward: bool = True,
    return_first: bool = False,
    min_count: int | None = 1,
    max_count: int | None = None,
    sender_id: int | None = None,
):
    if isinstance(msg_or_event, MessageEvent):
        msg = get_msg(msg_or_event)
        event = msg_or_event
        sender_id = event.user_id
    else:
        msg = msg_or_event
        event = None

    if event and int(bot.self_id) == int(sender_id or 0):
        cqs = extract_cq_code(msg)
        if cqs.get("json"):
            raise ReplyException("暂时无法读取Bot发送的折叠消息中的图片，可以先手动转发该消息")

    ret = extract_image_data(msg)
    if parse_forward:
        cqs = extract_cq_code(msg)
        if "forward" in cqs:
            forward_id = cqs["forward"][0].get("id")
            if forward_id:
                forward_msg = await get_forward_msg(bot, forward_id)
                for msg_obj in forward_msg.get("messages") or []:
                    ret.extend(extract_image_data(msg_obj.get("message")))
    if parse_reply and event:
        reply_msg = await aget_reply_msg(bot, event)
        if reply_msg:
            reply_sender = None
            reply = event.get("reply")
            if isinstance(reply, dict):
                sender = reply.get("sender") or {}
                reply_sender = sender.get("user_id") if isinstance(sender, dict) else None
            ret.extend(
                await get_image_datas_from_msg(
                    bot,
                    reply_msg,
                    parse_reply=False,
                    parse_forward=parse_forward,
                    return_first=False,
                    min_count=None,
                    max_count=None,
                    sender_id=reply_sender,
                )
            )

    sources = "消息本身"
    if parse_forward:
        sources += "/折叠消息"
    if parse_reply:
        sources += "/回复消息"

    if return_first:
        if not ret:
            raise ReplyException(f"该指令需要输入一张图片，在{sources}中没有找到图片")
        return ret[0]
    if min_count:
        if len(ret) < min_count:
            raise ReplyException(f"该指令至少输入{min_count}张图片，在{sources}中仅找到{len(ret)}张图片")
    if max_count is not None and len(ret) > max_count:
        raise ReplyException(f"该指令最多输入{max_count}张图片，在{sources}中找到{len(ret)}张图片")
    return ret


async def get_image_urls_from_msg(
    bot: Bot,
    msg_or_event,
    parse_reply: bool = True,
    parse_forward: bool = True,
    return_first: bool = False,
    min_count: int | None = 1,
    max_count: int | None = None,
):
    ret = await get_image_datas_from_msg(
        bot,
        msg_or_event,
        parse_reply=parse_reply,
        parse_forward=parse_forward,
        return_first=return_first,
        min_count=min_count,
        max_count=max_count,
    )
    if return_first:
        return ret.get("url")
    return [item.get("url") for item in ret if item.get("url")]


async def get_avatar_url(bot: Bot | None, user_id: int) -> str:
    if bot and int(user_id) >= 10**10:
        try:
            return await bot.call_api("get_avatar_url", user_id=int(user_id))
        except Exception:
            pass
    return f"http://q1.qlogo.cn/g?b=qq&nk={int(user_id)}&s=100"


async def get_avatar_url_large(bot: Bot | None, user_id: int) -> str:
    if bot and int(user_id) >= 10**10:
        try:
            return await bot.call_api("get_avatar_url", user_id=int(user_id))
        except Exception:
            pass
    return f"http://q1.qlogo.cn/g?b=qq&nk={int(user_id)}&s=640"


def get_user_name_by_event(event_or_reply) -> str:
    sender = getattr(event_or_reply, "sender", None)
    if sender is None and isinstance(event_or_reply, dict):
        sender = event_or_reply.get("sender")
    if isinstance(sender, dict):
        return str(sender.get("card") or sender.get("nickname") or event_or_reply.get("user_id") or "")
    if sender is not None:
        return str(getattr(sender, "card", None) or getattr(sender, "nickname", None) or getattr(event_or_reply, "user_id", "") or "")
    return str(getattr(event_or_reply, "user_id", "") or "")


def _remember_reply(ret: Any) -> None:
    try:
        if ret and ret.get("message_id"):
            msg_id = int(ret["message_id"])
            _bot_reply_msg_ids.add(msg_id)
            remember_logged_msg(msg_id)
    except Exception:
        pass


def _resolve_bot(bot: Bot | None = None) -> Bot:
    return bot if bot is not None else get_bot()


def _group_label(group_id: int | None) -> str:
    gid = int(group_id or 0)
    if not gid:
        return ""
    name = get_cached_group_name(gid)
    return f"{name}({gid})" if name else str(gid)


async def get_group_name(bot: Bot, group_id: int) -> str:
    gid = int(group_id)
    name = get_cached_group_name(gid)
    if name:
        return name
    try:
        info = await bot.get_group_info(gid)
        if isinstance(info, dict):
            name = str(info.get("group_name") or "")
            if name:
                cache_group_name(gid, name)
                return name
    except Exception:
        logger.debug("获取群名失败 group=%s", gid, exc_info=True)
    return str(gid)


def _log_outbound(
    bot: Bot,
    message: MessageLike,
    *,
    group_id: int | None = None,
    user_id: int | None = None,
    ret: Any = None,
    kind: str = "",
) -> None:
    try:
        msg_id = 0
        if isinstance(ret, dict) and ret.get("message_id"):
            msg_id = int(ret["message_id"])
        summary = summarize_message(Message.of(message))
        if kind == "fold":
            summary = f"[转发] {summary}"
        if group_id:
            core_logger.info("[%s] %s 自身消息: %s", msg_id or "-", _group_label(group_id), summary)
        else:
            core_logger.info("[%s] 自身私聊 %s: %s", msg_id or "-", user_id or bot.self_id, summary)
    except Exception:
        logger.debug("记录出站消息失败", exc_info=True)


async def _member_name(bot: Bot, group_id: int | None, user_id: int) -> str:
    uid = int(user_id)
    if group_id:
        key = (int(group_id), uid)
        cached = _group_member_name_cache.get(key)
        if cached:
            return cached
        try:
            info = await bot.get_group_member_info(int(group_id), uid)
            if isinstance(info, dict):
                name = str(info.get("card") or info.get("nickname") or uid)
                _group_member_name_cache.set(key, name)
                return name
        except Exception:
            logger.debug("获取群成员名失败 group=%s user=%s", group_id, uid, exc_info=True)
    cached = _stranger_name_cache.get(uid)
    if cached:
        return cached
    try:
        info = await bot.get_stranger_info(uid)
        if isinstance(info, dict):
            name = str(info.get("nickname") or uid)
            _stranger_name_cache.set(uid, name)
            return name
    except Exception:
        logger.debug("获取用户名失败 user=%s", uid, exc_info=True)
    return str(uid)


async def get_group_member_name(group_id: int, user_id: int, bot: Bot | None = None) -> str:
    """调用API获取群聊中的用户名（带缓存），有群名片则返回群名片，否则返回昵称。"""
    return await _member_name(_resolve_bot(bot), group_id, user_id)


# ============================ 出站消息 ============================ #


async def send_msg(bot: Bot, event: MessageEvent, message: MessageLike) -> Any:
    ret = await bot.send(event, message)
    _remember_reply(ret)
    _log_outbound(
        bot,
        message,
        group_id=event.group_id if is_group_msg(event) else None,
        user_id=event.user_id,
        ret=ret,
    )
    return ret


async def send_reply_msg(bot: Bot, event: MessageEvent, message: MessageLike) -> Any:
    ret = await bot.send(event, message, reply=True)
    _remember_reply(ret)
    logged = Message.of(message)
    if event.message_id:
        logged = Message.of(f"[CQ:reply,id={event.message_id}]") + logged
    _log_outbound(
        bot,
        logged,
        group_id=event.group_id if is_group_msg(event) else None,
        user_id=event.user_id,
        ret=ret,
    )
    return ret


async def send_at_msg(bot: Bot, event: MessageEvent, message: MessageLike) -> Any:
    ret = await bot.send(event, message, at_sender=True)
    _remember_reply(ret)
    logged = Message.of(Segment.at(event.user_id)) + Message.of(" ") + Message.of(message)
    _log_outbound(
        bot,
        logged,
        group_id=event.group_id if is_group_msg(event) else None,
        user_id=event.user_id,
        ret=ret,
    )
    return ret


async def send_group_msg_by_bot(group_id: int, content: MessageLike, bot: Bot | None = None) -> Any:
    bot = _resolve_bot(bot)
    ret = await bot.send_group_msg(int(group_id), content)
    _remember_reply(ret)
    _log_outbound(bot, content, group_id=int(group_id), ret=ret)
    return ret


async def send_private_msg_by_bot(user_id: int, content: MessageLike, bot: Bot | None = None) -> Any:
    bot = _resolve_bot(bot)
    ret = await bot.send_private_msg(int(user_id), content)
    _remember_reply(ret)
    _log_outbound(bot, content, user_id=int(user_id), ret=ret)
    return ret


@dataclass
class FoldMsgPart:
    type: str
    content: str

    def get_linecount(self) -> int:
        if self.type == "text":
            ret = 0
            for part in self.content.split("\n"):
                ret += get_str_display_length(part) // 40 + 1
            return ret
        return 4

    def get_text_length(self) -> int:
        if self.type == "text":
            return len(self.content)
        return 0


def contents_to_parts(contents: str | list[str]) -> list[list[FoldMsgPart]]:
    if isinstance(contents, str):
        contents = [contents]
    ret: list[list[FoldMsgPart]] = []
    for content in contents:
        parts: list[FoldMsgPart] = []
        cur = 0
        while cur < len(content):
            cq_start = content.find("[CQ:", cur)
            if cq_start == -1:
                if cur < len(content):
                    parts.append(FoldMsgPart("text", content[cur:]))
                break
            if cur < cq_start:
                parts.append(FoldMsgPart("text", content[cur:cq_start]))
            cq_end = content.find("]", cq_start)
            cq_type_end = content.find(",", cq_start)
            if cq_end == -1:
                parts.append(FoldMsgPart("text", content[cq_start:]))
                break
            if cq_type_end == -1 or cq_type_end > cq_end:
                cq_type_end = cq_end
            parts.append(
                FoldMsgPart(
                    content[cq_start + 4 : cq_type_end],
                    content[cq_start : cq_end + 1],
                )
            )
            cur = cq_end + 1
        ret.append(parts)
    return ret


def parts_to_contents(parts: list[list[FoldMsgPart]]) -> list[str]:
    return ["".join(part.content for part in part_list) for part_list in parts]


def apply_limit_to_parts(fold_parts: list[list[FoldMsgPart]], limit: int, seq: int = 1) -> list[list[list[FoldMsgPart]]]:
    max_seg = int(get_cfg_or_value(MAX_FOLD_MSG_SEGMENT_COUNT_CFG, 5) or 5)
    if seq > max_seg:
        raise ValueError(f"折叠消息分段超过最大限制 {max_seg}")
    if limit <= 64:
        limit = 65
    if not fold_parts:
        return []
    if seq > 1:
        fold_parts = [[FoldMsgPart("limit_text", f"[分段折叠消息Part.{seq}]")]] + fold_parts
    cur_len = 0
    cur_fold: list[list[FoldMsgPart]] = []
    for i, msg in enumerate(fold_parts):
        msg_len = sum(part.get_text_length() for part in msg)
        cur_msg: list[FoldMsgPart] = []
        if msg_len > limit:
            rest_limit, parts_len = limit - cur_len, 0
            for part in msg:
                part_len = part.get_text_length()
                if part.type == "text" and parts_len + part_len > rest_limit:
                    part1 = FoldMsgPart(part.type, part.content[: rest_limit - parts_len])
                    part2 = FoldMsgPart(part.type, part.content[rest_limit - parts_len :])
                    cur_msg.append(part1)
                    cur_msg = [p for p in cur_msg if p.content]
                    cur_fold.append(cur_msg)
                    cur_fold.append([FoldMsgPart("limit_text", "[消息过长已自动分段]")])
                    cur_fold = [m for m in cur_fold if m]
                    rest_fold_parts = [[part2]] + fold_parts[i + 1 :]
                    return [cur_fold] + apply_limit_to_parts(rest_fold_parts, limit, seq + 1)
                cur_msg.append(part)
                parts_len += part_len
        elif cur_len + msg_len > limit:
            cur_fold.append([FoldMsgPart("limit_text", "[消息过长已自动分段]")])
            return [cur_fold] + apply_limit_to_parts(fold_parts[i:], limit, seq + 1)
        else:
            cur_msg = msg
        cur_msg = [p for p in cur_msg if p.content]
        cur_fold.append(cur_msg)
        cur_len += sum(part.get_text_length() for part in cur_msg)
    cur_fold = [m for m in cur_fold if m]
    return [cur_fold] if cur_fold else []


async def fold_msg_fallback(
    bot: Bot,
    group_id: int | None,
    user_id: int | None,
    contents: list[str],
    e: Exception,
    method: str | None,
) -> Any:
    if method is None:
        method = str(get_cfg_or_value(DEFAULT_FOLD_FALLBACK_METHOD_CFG, "join_newline") or "join_newline")

    async def send(msg: str):
        if group_id:
            return await send_group_msg_by_bot(group_id, msg, bot=bot)
        if user_id:
            return await send_private_msg_by_bot(user_id, msg, bot=bot)
        raise ValueError("折叠消息 fallback 缺少 group_id / user_id")

    logger.warning("发送折叠消息失败，fallback=%s: %s", method, get_exc_desc(e))
    if method == "seperate":
        contents = list(contents)
        contents[0] = "（发送折叠消息失败）\n" + contents[0]
        ret = None
        for content in contents:
            ret = await send(content)
        return ret
    if method == "join_newline":
        return await send("\n".join(["（发送折叠消息失败）"] + list(contents)))
    if method == "join":
        return await send("".join(["（发送折叠消息失败）\n"] + list(contents)))
    if method == "none":
        return await send("发送折叠消息失败")
    raise Exception(f"未知折叠消息 fallback 方法 {method}")


async def send_fold_msg(
    bot: Bot,
    group_id: int | None,
    user_id: int | None,
    contents: str | list[str],
    fallback_method: str | None = None,
    first_is_user: bool = False,
) -> Any:
    parts = contents_to_parts(contents)
    if not parts:
        raise ValueError("发送的折叠消息为空")
    all_parts = apply_limit_to_parts(parts, int(get_cfg_or_value(MAX_FOLD_MSG_LEN_CFG, 6000) or 6000))
    ret = None
    for fold_parts in all_parts:
        contents_list = parts_to_contents(fold_parts)
        selfname = await _member_name(bot, group_id, bot.self_id)
        msg_list = []
        for i, content in enumerate(contents_list):
            if i == 0 and first_is_user and user_id:
                uid = int(user_id)
                nickname = await _member_name(bot, group_id, user_id)
            else:
                uid = int(bot.self_id)
                nickname = selfname
            msg_list.append(
                {
                    "type": "node",
                    "data": {
                        "user_id": uid,
                        "nickname": nickname,
                        "content": content if content else "（无文字内容）",
                    },
                }
            )
        if not msg_list:
            msg_list.append(
                {
                    "type": "node",
                    "data": {
                        "user_id": int(bot.self_id),
                        "nickname": selfname,
                        "content": "数据为空，无法折叠",
                    },
                }
            )
        try:
            if group_id:
                ret = await bot.send_group_forward_msg(group_id=int(group_id), messages=msg_list)
            else:
                if not user_id:
                    raise ValueError("私聊折叠消息缺少 user_id")
                ret = await bot.send_private_forward_msg(user_id=int(user_id), messages=msg_list)
            _remember_reply(ret)
            preview = " | ".join(c.replace("\n", " ") for c in contents_list if c)[:160] or "[转发]"
            _log_outbound(bot, preview, group_id=group_id, user_id=user_id, ret=ret, kind="fold")
        except Exception as e:
            ret = await fold_msg_fallback(bot, group_id, user_id, contents_list, e, fallback_method)
    return ret


async def send_fold_msg_adaptive(
    bot: Bot,
    group_id: int | None,
    user_id: int | None,
    contents: str | list[str],
    not_fold_contents: str | list[str] | None = None,
    threshold: Any = DEFAULT_FOLD_THRESHOLD_CFG,
    need_reply: bool = True,
    reply_message_id: int | None = None,
    fallback_method: str | None = None,
    first_is_user: bool = False,
) -> Any:
    if isinstance(contents, str):
        contents = [contents]
    all_parts = contents_to_parts(contents)
    linecount = len(all_parts) - 1
    for parts in all_parts:
        for part in parts:
            linecount += part.get_linecount()
    logger.debug("折叠消息行数: %s", linecount)
    if linecount < int(get_cfg_or_value(threshold, 20) or 20):
        if not_fold_contents is not None:
            if isinstance(not_fold_contents, str):
                not_fold_contents = [not_fold_contents]
            contents = not_fold_contents
        reply_cq = ""
        if need_reply:
            if reply_message_id is None:
                raise ValueError("需要回复消息时 reply_message_id 不能为空")
            reply_cq = f"[CQ:reply,id={reply_message_id}]"
        ret = None
        for content in contents:
            msg = f"{reply_cq}{content}"
            if group_id:
                ret = await send_group_msg_by_bot(group_id, msg, bot=bot)
            else:
                if not user_id:
                    raise ValueError("私聊消息缺少 user_id")
                ret = await send_private_msg_by_bot(user_id, msg, bot=bot)
        return ret
    return await send_fold_msg(
        bot,
        group_id,
        user_id,
        contents,
        fallback_method=fallback_method,
        first_is_user=first_is_user,
    )


# ============================ 服务开关 ============================ #


class GroupWhiteList:
    """群白名单：默认关闭。始终注册 /{name} on|off|status；is_service=True 时才会进入 /服务。"""

    def __init__(self, db, logger_, name: str, superuser=None, on_func=None, off_func=None) -> None:
        self.db = db
        self.logger = logger_
        self.name = name
        self.superuser = superuser
        self.white_list_name = f"group_white_list_{name}"
        self.on_func = on_func
        self.off_func = off_func

    def _register_switch_cmds(self) -> None:
        async def get_group_id_desc(ctx: HandlerContext) -> tuple[int, str]:
            args = ctx.get_args().strip()
            if args:
                try:
                    group_id = int(args.split()[0])
                except (TypeError, ValueError):
                    raise ReplyException(f"无效群聊 {args}")
                groups = await ctx.bot.get_group_list()
                group = find_by(groups or [], "group_id", group_id)
                if group is None:
                    group = find_by(groups or [], "group_id", str(group_id))
                assert_and_reply(group, f"无效群聊 {args}")
                group_id = int(group["group_id"])
                name = str(group.get("group_name") or "")
                if name:
                    cache_group_name(group_id, name)
                group_desc = f'"{name}"({group_id})' if name else str(group_id)
            else:
                assert_and_reply(ctx.group_id, "请在群聊中使用，或指定群号")
                group_id = int(ctx.group_id)
                group_desc = "本群"
            return group_id, group_desc

        switch_on = CmdHandler([f"/{self.name} on"], logger, help_command="/{服务名} on", disable_help=True, priority=10)
        switch_on.check_superuser(self.superuser)

        @switch_on.handle()
        async def _(ctx: HandlerContext):
            group_id, group_desc = await get_group_id_desc(ctx)
            white_list = list(self.db.get(self.white_list_name, []))
            if group_id in white_list or str(group_id) in white_list:
                return await ctx.asend_reply_msg(f"{group_desc}的{self.name}已经是开启状态")
            white_list.append(group_id)
            self.db.set(self.white_list_name, white_list)
            if self.on_func is not None:
                ret = self.on_func(group_id)
                if hasattr(ret, "__await__"):
                    await ret
            return await ctx.asend_reply_msg(f"成功开启{group_desc}的{self.name}")

        switch_off = CmdHandler([f"/{self.name} off"], logger, help_command="/{服务名} off", disable_help=True, priority=10)
        switch_off.check_superuser(self.superuser)

        @switch_off.handle()
        async def _(ctx: HandlerContext):
            group_id, group_desc = await get_group_id_desc(ctx)
            white_list = list(self.db.get(self.white_list_name, []))
            if group_id not in white_list and str(group_id) not in white_list:
                return await ctx.asend_reply_msg(f"{group_desc}的{self.name}已经是关闭状态")
            self.remove(group_id)
            if self.off_func is not None:
                ret = self.off_func(group_id)
                if hasattr(ret, "__await__"):
                    await ret
            return await ctx.asend_reply_msg(f"成功关闭{group_desc}的{self.name}")

        switch_query = CmdHandler([f"/{self.name} status"], logger, help_command="/{服务名} status", disable_help=True, priority=10)

        @switch_query.handle()
        async def _(ctx: HandlerContext):
            group_id, group_desc = await get_group_id_desc(ctx)
            if self.check_id(group_id):
                return await ctx.asend_reply_msg(f"{group_desc}的{self.name}开启中")
            return await ctx.asend_reply_msg(f"{group_desc}的{self.name}关闭中")

    def get(self) -> list[int]:
        return [int(x) for x in self.db.get(self.white_list_name, [])]

    def add(self, group_id: int) -> bool:
        group_id = int(group_id)
        white_list = list(self.db.get(self.white_list_name, []))
        if group_id in white_list or str(group_id) in white_list:
            return False
        white_list.append(group_id)
        self.db.set(self.white_list_name, white_list)
        self.logger.info("添加群 %s 到 %s", group_id, self.white_list_name)
        if self.on_func is not None:
            self.on_func(group_id)
        return True

    def remove(self, group_id: int) -> bool:
        group_id = int(group_id)
        white_list = list(self.db.get(self.white_list_name, []))
        if group_id not in white_list and str(group_id) not in white_list:
            return False
        white_list = [x for x in white_list if int(x) != group_id]
        self.db.set(self.white_list_name, white_list)
        self.logger.info("从 %s 删除群 %s", self.white_list_name, group_id)
        if self.off_func is not None:
            self.off_func(group_id)
        return True

    def check_id(self, group_id: int) -> bool:
        return int(group_id) in [int(x) for x in self.get()]

    def check(self, event: MessageEvent, allow_private=False, allow_super=True) -> bool:
        if is_group_msg(event):
            if allow_super and check_superuser(event, self.superuser):
                return True
            return self.check_id(event.group_id)
        return allow_private


class GroupBlackList:
    """群黑名单：默认开启。始终注册 /{name} on|off|status；is_service=True 时才会进入 /服务。"""

    def __init__(self, db, logger_, name: str, superuser=None, on_func=None, off_func=None) -> None:
        self.db = db
        self.logger = logger_
        self.name = name
        self.superuser = superuser
        self.black_list_name = f"group_black_list_{name}"
        self.on_func = on_func
        self.off_func = off_func

    def _register_switch_cmds(self) -> None:
        async def get_group_id_desc(ctx: HandlerContext) -> tuple[int, str]:
            args = ctx.get_args().strip()
            if args:
                try:
                    group_id = int(args.split()[0])
                except (TypeError, ValueError):
                    raise ReplyException(f"无效群聊 {args}")
                groups = await ctx.bot.get_group_list()
                group = find_by(groups or [], "group_id", group_id)
                if group is None:
                    group = find_by(groups or [], "group_id", str(group_id))
                assert_and_reply(group, f"无效群聊 {args}")
                group_id = int(group["group_id"])
                name = str(group.get("group_name") or "")
                if name:
                    cache_group_name(group_id, name)
                group_desc = f'"{name}"({group_id})' if name else str(group_id)
            else:
                assert_and_reply(ctx.group_id, "请在群聊中使用，或指定群号")
                group_id = int(ctx.group_id)
                group_desc = "本群"
            return group_id, group_desc

        switch_off = CmdHandler([f"/{self.name} off"], logger, help_command="/{服务名} off", disable_help=True, priority=10)
        switch_off.check_superuser(self.superuser)

        @switch_off.handle()
        async def _(ctx: HandlerContext):
            group_id, group_desc = await get_group_id_desc(ctx)
            if not self.check_id(group_id):
                return await ctx.asend_reply_msg(f"{group_desc}的{self.name}已经是关闭状态")
            self.add(group_id)
            if self.off_func is not None:
                ret = self.off_func(group_id)
                if hasattr(ret, "__await__"):
                    await ret
            return await ctx.asend_reply_msg(f"{group_desc}的{self.name}已关闭")

        switch_on = CmdHandler([f"/{self.name} on"], logger, help_command="/{服务名} on", disable_help=True, priority=10)
        switch_on.check_superuser(self.superuser)

        @switch_on.handle()
        async def _(ctx: HandlerContext):
            group_id, group_desc = await get_group_id_desc(ctx)
            if self.check_id(group_id):
                return await ctx.asend_reply_msg(f"{group_desc}的{self.name}已经是开启状态")
            self.remove(group_id)
            if self.on_func is not None:
                ret = self.on_func(group_id)
                if hasattr(ret, "__await__"):
                    await ret
            return await ctx.asend_reply_msg(f"{group_desc}的{self.name}已开启")

        switch_query = CmdHandler([f"/{self.name} status"], logger, help_command="/{服务名} status", disable_help=True, priority=10)

        @switch_query.handle()
        async def _(ctx: HandlerContext):
            group_id, group_desc = await get_group_id_desc(ctx)
            if self.check_id(group_id):
                return await ctx.asend_reply_msg(f"{group_desc}的{self.name}开启中")
            return await ctx.asend_reply_msg(f"{group_desc}的{self.name}关闭中")

    def get(self) -> list[int]:
        return [int(x) for x in self.db.get(self.black_list_name, [])]

    def add(self, group_id: int) -> bool:
        group_id = int(group_id)
        black_list = list(self.db.get(self.black_list_name, []))
        if group_id in black_list or str(group_id) in black_list:
            return False
        black_list.append(group_id)
        self.db.set(self.black_list_name, black_list)
        self.logger.info("添加群 %s 到 %s", group_id, self.black_list_name)
        if self.off_func is not None:
            self.off_func(group_id)
        return True

    def remove(self, group_id: int) -> bool:
        group_id = int(group_id)
        black_list = list(self.db.get(self.black_list_name, []))
        if group_id not in black_list and str(group_id) not in black_list:
            return False
        black_list = [x for x in black_list if int(x) != group_id]
        self.db.set(self.black_list_name, black_list)
        self.logger.info("从 %s 删除群 %s", self.black_list_name, group_id)
        if self.on_func is not None:
            self.on_func(group_id)
        return True

    def check_id(self, group_id) -> bool:
        return int(group_id) not in [int(x) for x in self.get()]

    def check(self, event: MessageEvent, allow_private=False, allow_super=True) -> bool:
        if is_group_msg(event):
            if allow_super and check_superuser(event, self.superuser):
                return True
            return self.check_id(event.group_id)
        return allow_private


_gwls: dict[str, GroupWhiteList] = {}
_gbls: dict[str, GroupBlackList] = {}
_gwl_all: dict[str, GroupWhiteList] = {}
_gbl_all: dict[str, GroupBlackList] = {}


def get_group_white_list(db, logger_, name: str, superuser=None, on_func=None, off_func=None, is_service=True):
    """is_service 只控制是否进入 /服务；开关指令和 db 键始终会有。"""
    inst = _gwl_all.get(name)
    if inst is None:
        inst = GroupWhiteList(db, logger_, name, superuser, on_func, off_func)
        inst._register_switch_cmds()
        _gwl_all[name] = inst
    if is_service:
        _gwls[name] = inst
    return inst


def get_group_black_list(db, logger_, name: str, superuser=None, on_func=None, off_func=None, is_service=True):
    """is_service 只控制是否进入 /服务；开关指令和 db 键始终会有。"""
    inst = _gbl_all.get(name)
    if inst is None:
        inst = GroupBlackList(db, logger_, name, superuser, on_func, off_func)
        inst._register_switch_cmds()
        _gbl_all[name] = inst
    if is_service:
        _gbls[name] = inst
    return inst


class ColdDown:
    def __init__(self, db, logger_, interval=None, cold_down_name: str = "cd") -> None:
        self.db = db
        self.logger = logger_
        self.interval = interval
        self.name = cold_down_name

    def _interval_seconds(self) -> float:
        if self.interval is None:
            return float(get_cfg_or_value(DEFAULT_CD_CFG, 2) or 0)
        return float(get_cfg_or_value(self.interval, 0) or 0)

    async def check(self, event: MessageEvent, allow_super=True, verbose=True) -> bool:
        if allow_super and check_superuser(event):
            return True
        seconds = self._interval_seconds()
        if seconds <= 0:
            return True
        key = f"{self.name}.{event.user_id}"
        last = float(self.db.get(key, 0) or 0)
        now = datetime.now().timestamp()
        if now - last < seconds:
            if verbose:
                verbose_key = f"{self.name}.verbose.{event.user_id}"
                last_verbose = float(self.db.get(verbose_key, 0) or 0)
                verbose_interval = float(get_cfg_or_value(CD_VERBOSE_INTERVAL_CFG, 30) or 30)
                if now - last_verbose > verbose_interval:
                    self.db.set(verbose_key, now)
                    rest = timedelta(seconds=max(0, seconds - (now - last)))
                    msg = f"冷却中, 剩余时间: {get_readable_timedelta(rest, precision='s')}"
                    try:
                        reply = f"[CQ:reply,id={event.message_id}] {msg}" if event.message_id else msg
                        if is_group_msg(event) and event.group_id:
                            await send_group_msg_by_bot(event.group_id, reply)
                        else:
                            await send_private_msg_by_bot(event.user_id, reply)
                    except Exception:
                        self.logger.print_exc(f"{self.name} 发送冷却提示失败")
            return False
        self.db.set(key, now)
        return True


@dataclass
class HandlerContext:
    time: datetime | None = None
    handler: Any = None
    bot: Bot | None = None
    event: MessageEvent | None = None
    trigger_cmd: str = ""
    arg_text: str = ""
    message_id: int = 0
    user_id: int = 0
    group_id: int = 0
    logger: Any = None
    block_ids: list[str] = field(default_factory=list)

    def get_args(self) -> str:
        return self.arg_text

    def get_msg(self) -> list[dict]:
        return get_msg(self.event)

    def get_reply_msg(self):
        return get_reply_msg(self.event)

    async def aget_reply_msg(self):
        return await aget_reply_msg(self.bot, self.event)

    def get_reply_msg_id(self) -> int | None:
        return get_reply_id(self.event)

    def get_reply_sender(self):
        reply = self.event.get("reply") if self.event else None
        if isinstance(reply, dict):
            return reply.get("sender")
        return getattr(reply, "sender", None)

    def get_at_qids(self) -> list[int]:
        return extract_at_qq(self.get_msg())

    def aget_image_datas(
        self,
        parse_reply: bool = True,
        parse_forward: bool = True,
        return_first: bool = False,
        min_count: int | None = 1,
        max_count: int | None = None,
    ):
        return get_image_datas_from_msg(
            self.bot,
            self.event,
            parse_reply=parse_reply,
            parse_forward=parse_forward,
            return_first=return_first,
            min_count=min_count,
            max_count=max_count,
        )

    def aget_image_urls(
        self,
        parse_reply: bool = True,
        parse_forward: bool = True,
        return_first: bool = False,
        min_count: int | None = 1,
        max_count: int | None = None,
    ):
        return get_image_urls_from_msg(
            self.bot,
            self.event,
            parse_reply=parse_reply,
            parse_forward=parse_forward,
            return_first=return_first,
            min_count=min_count,
            max_count=max_count,
        )

    async def block(self, block_id: str = "", timeout: int = 3 * 60, err_msg: str | None = None):
        block_id = str(block_id)
        start = datetime.now()
        while True:
            if block_id not in self.handler.block_set:
                break
            if (datetime.now() - start).seconds > timeout:
                if err_msg is None:
                    err_msg = f"指令执行繁忙(block_id={block_id})，请稍后再试"
                raise ReplyException(err_msg)
            await asyncio.sleep(1)
        self.handler.block_set.add(block_id)
        self.block_ids.append(block_id)

    async def asend_msg(self, msg: MessageLike):
        return await send_msg(self.bot, self.event, msg)

    async def asend_reply_msg(self, msg: MessageLike):
        return await send_reply_msg(self.bot, self.event, msg)

    async def asend_at_msg(self, msg: MessageLike):
        return await send_at_msg(self.bot, self.event, msg)

    async def asend_fold_msg(
        self,
        contents: str | list[str],
        show_command: bool = True,
        fallback_method: str | None = None,
    ):
        if isinstance(contents, str):
            contents = [contents]
        first_is_user = False
        if show_command and self.event is not None:
            contents = [self.event.plain_text] + contents
            first_is_user = True
        return await send_fold_msg(
            bot=self.bot,
            group_id=self.group_id or None,
            user_id=self.user_id,
            contents=contents,
            fallback_method=fallback_method,
            first_is_user=first_is_user,
        )

    async def asend_fold_msg_adaptive(
        self,
        contents: str | list[str],
        threshold=None,
        need_reply: bool = True,
        fallback_method: str | None = None,
    ):
        if isinstance(contents, str):
            contents = [contents]
        fold_contents = contents
        first_is_user = False
        if need_reply and self.event is not None:
            fold_contents = [self.event.plain_text] + contents
            first_is_user = True
        return await send_fold_msg_adaptive(
            bot=self.bot,
            group_id=self.group_id or None,
            user_id=self.user_id,
            contents=fold_contents,
            not_fold_contents=contents,
            threshold=threshold if threshold is not None else DEFAULT_FOLD_THRESHOLD_CFG,
            need_reply=need_reply,
            reply_message_id=self.message_id,
            fallback_method=fallback_method,
            first_is_user=first_is_user,
        )


_cmd_history: list[HandlerContext] = []
MAX_CMD_HISTORY = 100
HELP_DOC_PATH = "helps/*.md"
HELP_PART_IMG_CACHE_DIR = "data/utils/help_part_img_cache/"
SEG_COMMAND_SEPS = ["", " ", "_"]


class SegCmd:
    """由多段构成的指令，展开成无分隔 / 空格 / 下划线三种写法。"""

    def __init__(self, *args, seps: list[str] = SEG_COMMAND_SEPS) -> None:
        self.commands: set[str] = set()
        assert args, "至少需要一个参数"
        if len(args) == 1:
            raw = args[0]
            for sep in SEG_COMMAND_SEPS:
                if sep:
                    raw = raw.replace(sep, " ")
            args = raw.split()
        for sep in seps:
            self.commands.add(sep.join(args))

    def get(self) -> list[str]:
        return list(self.commands)


@dataclass
class HelpDocCmdPart:
    doc_name: str
    cmds: set[str] | None = None
    content: str = ""
    md5: str = ""


@dataclass
class HelpDoc:
    mtime: int
    parts: list[HelpDocCmdPart] = field(default_factory=list)


class CmdHandler:
    cmd_handlers: list["CmdHandler"] = []
    help_docs: dict[str, HelpDoc] = {}

    def __init__(
        self,
        commands,
        logger_,
        error_reply=True,
        priority=0,
        block=True,
        only_to_me=False,
        disabled=False,
        banned_cmds=None,
        check_group_enabled=True,
        allow_bot_reply_msg=False,
        help_command: str | None = None,
        disable_help=False,
        help_trigger_condition="exact",
        use_seg_cmd=True,
    ) -> None:
        if isinstance(commands, str) or isinstance(commands, SegCmd):
            commands = [commands]
        self.commands = []
        for cmd in commands:
            if isinstance(cmd, SegCmd):
                self.commands.extend(cmd.get())
            elif cmd == "":
                self.commands.append("")
            elif use_seg_cmd:
                self.commands.extend(SegCmd(str(cmd)).get())
            else:
                self.commands.append(str(cmd))
        self.commands = list(set(self.commands))
        self.commands.sort(key=len, reverse=True)
        self.logger = logger_
        self.error_reply = error_reply
        self.priority = priority
        self.block = block
        self.only_to_me = only_to_me
        self.disabled = disabled
        self.banned_cmds = [banned_cmds] if isinstance(banned_cmds, str) else (banned_cmds or [])
        self.check_group_enabled = check_group_enabled
        self.allow_bot_reply_msg = allow_bot_reply_msg
        self.help_command = help_command
        self.disable_help = disable_help
        self.help_trigger_condition = help_trigger_condition
        self.superuser_check = None
        self.private_group_check = None
        self.wblist_checks = []
        self.cdrate_checks = []
        self.handler_func = None
        self.block_set: set[str] = set()
        CmdHandler.cmd_handlers.append(self)
        CmdHandler.cmd_handlers.sort(
            key=lambda x: (x.priority, max((len(c) for c in x.commands), default=0)),
            reverse=True,
        )

    def check_group(self):
        self.private_group_check = "group"
        return self

    def check_private(self):
        self.private_group_check = "private"
        return self

    def check_wblist(self, wblist, allow_private=True, allow_super=True):
        self.wblist_checks.append((wblist, {"allow_private": allow_private, "allow_super": allow_super}))
        return self

    def check_cdrate(self, cd_rate, allow_super=True, verbose=True):
        self.cdrate_checks.append((cd_rate, {"allow_super": allow_super, "verbose": verbose}))
        return self

    def check_superuser(self, superuser=None):
        self.superuser_check = {"superuser": superuser}
        return self

    @classmethod
    def update_help_docs(cls) -> None:
        paths = list(glob.glob(HELP_DOC_PATH))
        names: set[str] = set()
        all_md5: set[str] = set()

        def parse_doc(path: str) -> list[HelpDocCmdPart]:
            help_doc = Path(path).read_text(encoding="utf-8")
            doc_name = Path(path).stem
            parts = help_doc.split("---")[2:-1]
            ret: list[HelpDocCmdPart] = []
            for part in parts:
                start = part.find("### ")
                if start == -1:
                    continue
                part = part[start:]
                for p in part.split("### "):
                    p = p.strip()
                    if p:
                        ret.append(
                            HelpDocCmdPart(
                                cmds=None,
                                doc_name=doc_name,
                                content="### " + p + f"\n\n>发送`/help {doc_name}`查看完整帮助",
                            )
                        )
            for part in ret:
                lines = part.content.splitlines()
                if len(lines) < 2:
                    continue
                start = lines[1].find("`")
                if start == -1:
                    continue
                part.cmds = set(lines[1][start:].replace("` `", "%").replace("`", "").strip().split("%"))
                part.md5 = get_md5(part.content)
            return ret

        for path in paths:
            try:
                name = Path(path).stem
                mtime = int(os.path.getmtime(path))
                if name not in cls.help_docs or cls.help_docs[name].mtime < mtime:
                    cls.help_docs[name] = HelpDoc(mtime=mtime)
                    cls.help_docs[name].parts = parse_doc(path)
                names.add(name)
            except Exception:
                logger.print_exc(f"解析帮助文档 {path} 失败")

        for name in list(cls.help_docs.keys()):
            if name not in names:
                del cls.help_docs[name]

        for doc in cls.help_docs.values():
            for part in doc.parts:
                if part.md5:
                    all_md5.add(part.md5)
        if os.path.isdir(HELP_PART_IMG_CACHE_DIR):
            for path in glob.glob(os.path.join(HELP_PART_IMG_CACHE_DIR, "*.png")):
                if Path(path).stem not in all_md5:
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.print_exc(f"删除帮助文档图片缓存 {path} 失败: {e}")

    @classmethod
    def find_cmd_help_doc(cls, cmd: str) -> HelpDocCmdPart | None:
        cls.update_help_docs()
        for doc in cls.help_docs.values():
            for part in doc.parts:
                if part.cmds and cmd in part.cmds:
                    return part
        return None

    @classmethod
    async def get_cmd_help_doc_img(cls, part: HelpDocCmdPart, width=640):
        from src.draw.markdown import markdown_to_image
        from src.draw.img_utils import open_image

        md5 = part.md5 or get_md5(part.content)
        cache_path = create_parent_folder(os.path.join(HELP_PART_IMG_CACHE_DIR, f"{md5}.png"))
        if os.path.exists(cache_path):
            return open_image(cache_path)
        img = await markdown_to_image(part.content, width=width)
        img.save(cache_path)
        return img

    def _match_help(self, arg_text: str) -> bool:
        if self.disable_help:
            return False
        cond = self.help_trigger_condition
        if isinstance(cond, str):
            for keyword in ("help", "帮助"):
                if cond == "contain" and keyword in arg_text:
                    return True
                if cond == "exact" and arg_text.strip() == keyword:
                    return True
            return False
        return bool(cond(arg_text))

    def handle(self):
        def decorator(handler_func):
            self.handler_func = handler_func
            return handler_func

        return decorator

    def match(self, text: str) -> tuple[str, str] | None:
        if not self.commands or "" in self.commands:
            return "", text
        for cmd in self.commands:
            if not cmd:
                continue
            if text == cmd or text.startswith(cmd):
                return cmd, text[len(cmd) :]
        return None

    async def dispatch(self, bot: Bot, event: MessageEvent) -> bool:
        if self.disabled or not self.handler_func:
            return False
        if event.is_sent:
            return False
        if not is_group_msg(event) and event.user_id == event.self_id:
            return False
        if not self.allow_bot_reply_msg and check_self_reply(event):
            return False

        plain = command_text(event)
        matched = self.match(plain)
        if matched is None:
            return False
        trigger_cmd, arg_text = matched
        if any(banned in trigger_cmd for banned in self.banned_cmds):
            return False
        if self.only_to_me and not event.to_me:
            if get_reply_id(event):
                await aget_reply_msg(bot, event)
            if not event.to_me:
                return False

        if self.private_group_check == "group" and not is_group_msg(event):
            return False
        if self.private_group_check == "private" and is_group_msg(event):
            return False
        if self.superuser_check and not check_superuser(event, **self.superuser_check):
            return False
        for wblist, kwargs in self.wblist_checks:
            if not wblist.check(event, **kwargs):
                return False
        for cdrate, kwargs in self.cdrate_checks:
            if not await cdrate.check(event, **kwargs):
                return False

        ctx = HandlerContext(
            time=datetime.now(),
            handler=self,
            bot=bot,
            event=event,
            trigger_cmd=trigger_cmd,
            arg_text=arg_text,
            message_id=event.message_id,
            user_id=event.user_id,
            group_id=event.group_id if is_group_msg(event) else 0,
            logger=self.logger,
        )
        if ctx.trigger_cmd:
            _cmd_history.append(ctx)
            overflow = len(_cmd_history) - MAX_CMD_HISTORY
            if overflow > 0:
                del _cmd_history[:overflow]
        try:
            if self._match_help(arg_text):
                cmds = self.commands if not self.help_command else [self.help_command]
                for cmd in cmds:
                    part = self.find_cmd_help_doc(cmd)
                    if part:
                        img = await self.get_cmd_help_doc_img(part)
                        await ctx.asend_reply_msg(await get_image_cq(img, low_quality=True))
                        break
                else:
                    raise ReplyException('没有找到该指令的帮助\n发送"/help"查看完整帮助')
            else:
                await self.handler_func(ctx)
        except NoReplyException:
            pass
        except ReplyException as e:
            await ctx.asend_reply_msg(str(e))
        except StopPropagation:
            raise
        except Exception as e:
            self.logger.print_exc(f'指令"{ctx.trigger_cmd}"处理失败')
            if self.error_reply:
                await ctx.asend_reply_msg(truncate(f"指令处理失败: {get_exc_desc(e)}", 256))
        finally:
            for block_id in ctx.block_ids:
                self.block_set.discard(block_id)
        if self.block:
            raise StopPropagation()
        return True


@on_message(priority=0)
async def _dispatch_commands(bot: Bot, event: MessageEvent):
    if not isinstance(event, MessageEvent) or event.is_sent:
        return
    # 跨 handler 走最长前缀，避免 /看 抢走 /看所有
    plain = command_text(event)
    ranked: list[tuple[int, int, CmdHandler]] = []
    for handler in CmdHandler.cmd_handlers:
        if handler.disabled or not handler.handler_func:
            continue
        matched = handler.match(plain)
        if matched is None:
            continue
        ranked.append((len(matched[0]), handler.priority, handler))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for _, _, handler in ranked:
        if await handler.dispatch(bot, event):
            if handler.block:
                return


def assert_and_reply(condition, msg: str):
    if not condition:
        raise ReplyException(msg)


_service_handler = CmdHandler(["/service", "/服务"], logger)
_service_handler.check_superuser()


@_service_handler.handle()
async def _(ctx: HandlerContext):
    name = ctx.get_args().strip()
    if name:
        assert_and_reply(name in _gwls or name in _gbls, f"未知服务 {name}")
        msg = ""
        if name in _gwls:
            msg += f"{name}使用的规则是白名单\n开启服务的群聊有:\n"
            for group_id in _gwls[name].get():
                msg += f"{await get_group_name(ctx.bot, group_id)}({group_id})\n"
        elif name in _gbls:
            msg += f"{name}使用的规则是黑名单\n关闭服务的群聊有:\n"
            for group_id in _gbls[name].get():
                msg += f"{await get_group_name(ctx.bot, group_id)}({group_id})\n"
        return await ctx.asend_reply_msg(msg.strip())

    assert_and_reply(ctx.group_id, "请在群聊中使用，或指定服务名")
    msg_on = "本群开启的服务:\n"
    msg_off = "本群关闭的服务:\n"
    for svc_name, gwl in _gwls.items():
        if gwl.check_id(ctx.group_id):
            msg_on += f"{svc_name} "
        else:
            msg_off += f"{svc_name} "
    for svc_name, gbl in _gbls.items():
        if gbl.check_id(ctx.group_id):
            msg_on += f"{svc_name} "
        else:
            msg_off += f"{svc_name} "
    return await ctx.asend_reply_msg(msg_on + "\n" + msg_off)