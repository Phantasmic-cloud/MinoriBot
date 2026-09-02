from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from .message import Message, Segment


def _as_int(v: Any, default: int = 0) -> int:
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


@dataclass
class Sender:
    user_id: int = 0
    nickname: str = ""
    card: str = ""
    sex: str = "unknown"
    age: int = 0
    area: str = ""
    level: str = ""
    role: str = "member"
    title: str = ""

    @classmethod
    def from_raw(cls, data: Any) -> Sender:
        if not isinstance(data, dict):
            return cls()
        return cls(
            user_id=_as_int(data.get("user_id")),
            nickname=_as_str(data.get("nickname")),
            card=_as_str(data.get("card")),
            sex=_as_str(data.get("sex"), "unknown"),
            age=_as_int(data.get("age")),
            area=_as_str(data.get("area")),
            level=_as_str(data.get("level")),
            role=_as_str(data.get("role"), "member"),
            title=_as_str(data.get("title")),
        )

    @property
    def display_name(self) -> str:
        return self.card or self.nickname or str(self.user_id)


@dataclass
class Anonymous:
    id: int = 0
    name: str = ""
    flag: str = ""

    @classmethod
    def from_raw(cls, data: Any) -> Anonymous | None:
        if not isinstance(data, dict):
            return None
        return cls(
            id=_as_int(data.get("id")),
            name=_as_str(data.get("name")),
            flag=_as_str(data.get("flag")),
        )


@dataclass
class Event:
    raw: dict[str, Any] = field(default_factory=dict)
    time: int = 0
    self_id: int = 0
    post_type: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def name(self) -> str:
        return self.post_type or "unknown"


@dataclass
class MessageEvent(Event):
    message_type: str = ""
    sub_type: str = ""
    message_id: int = 0
    user_id: int = 0
    message: Message = field(default_factory=Message)
    raw_message: str = ""
    font: int = 0
    sender: Sender = field(default_factory=Sender)
    to_me: bool = False
    group_id: int = 0
    anonymous: Anonymous | None = None
    target_id: int = 0
    temp_source: int = 0

    @property
    def is_group(self) -> bool:
        return self.message_type == "group"

    @property
    def is_private(self) -> bool:
        return self.message_type == "private"

    @property
    def is_sent(self) -> bool:
        return self.post_type == "message_sent"

    @property
    def plain_text(self) -> str:
        return self.message.plain_text

    @property
    def name(self) -> str:
        if self.message_type:
            return f"message.{self.message_type}"
        return "message"


@dataclass
class NoticeEvent(Event):
    notice_type: str = ""
    sub_type: str = ""
    user_id: int = 0
    group_id: int = 0
    operator_id: int = 0
    message_id: int = 0
    file: dict[str, Any] = field(default_factory=dict)
    flag: str = ""
    duration: int = 0
    honor_type: str = ""
    target_id: int = 0
    comment: str = ""
    likes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        if self.sub_type:
            return f"notice.{self.notice_type}.{self.sub_type}"
        if self.notice_type:
            return f"notice.{self.notice_type}"
        return "notice"


@dataclass
class RequestEvent(Event):
    request_type: str = ""
    sub_type: str = ""
    user_id: int = 0
    group_id: int = 0
    comment: str = ""
    flag: str = ""

    @property
    def name(self) -> str:
        if self.sub_type:
            return f"request.{self.request_type}.{self.sub_type}"
        if self.request_type:
            return f"request.{self.request_type}"
        return "request"


@dataclass
class MetaEvent(Event):
    meta_event_type: str = ""
    sub_type: str = ""
    status: dict[str, Any] = field(default_factory=dict)
    interval: int = 0

    @property
    def name(self) -> str:
        if self.sub_type:
            return f"meta_event.{self.meta_event_type}.{self.sub_type}"
        if self.meta_event_type:
            return f"meta_event.{self.meta_event_type}"
        return "meta_event"


def parse_event(raw: dict[str, Any]) -> Event:
    post_type = _as_str(raw.get("post_type"))
    if post_type in ("message", "message_sent"):
        return _parse_message(raw, post_type)
    if post_type == "notice":
        return _parse_notice(raw)
    if post_type == "request":
        return _parse_request(raw)
    if post_type == "meta_event":
        return _parse_meta(raw)
    return Event(
        raw=raw,
        time=_as_int(raw.get("time")),
        self_id=_as_int(raw.get("self_id")),
        post_type=post_type,
    )


def _parse_message(raw: dict[str, Any], post_type: str) -> MessageEvent:
    message = _parse_message_field(raw.get("message"), raw.get("raw_message"))
    event = MessageEvent(
        raw=raw,
        time=_as_int(raw.get("time")),
        self_id=_as_int(raw.get("self_id")),
        post_type=post_type,
        message_type=_as_str(raw.get("message_type")),
        sub_type=_as_str(raw.get("sub_type")),
        message_id=_as_int(raw.get("message_id")),
        user_id=_as_int(raw.get("user_id")),
        message=message,
        raw_message=_as_str(raw.get("raw_message")) or message.to_cq(),
        font=_as_int(raw.get("font")),
        sender=Sender.from_raw(raw.get("sender")),
        group_id=_as_int(raw.get("group_id")),
        anonymous=Anonymous.from_raw(raw.get("anonymous")),
        target_id=_as_int(raw.get("target_id")),
        temp_source=_as_int(raw.get("temp_source")),
    )
    event.to_me = _check_to_me(event)
    return event


def _parse_notice(raw: dict[str, Any]) -> NoticeEvent:
    return NoticeEvent(
        raw=raw,
        time=_as_int(raw.get("time")),
        self_id=_as_int(raw.get("self_id")),
        post_type="notice",
        notice_type=_as_str(raw.get("notice_type")),
        sub_type=_as_str(raw.get("sub_type")),
        user_id=_as_int(raw.get("user_id")),
        group_id=_as_int(raw.get("group_id")),
        operator_id=_as_int(raw.get("operator_id")),
        message_id=_as_int(raw.get("message_id")),
        file=raw.get("file") if isinstance(raw.get("file"), dict) else {},
        flag=_as_str(raw.get("flag")),
        duration=_as_int(raw.get("duration")),
        honor_type=_as_str(raw.get("honor_type")),
        target_id=_as_int(raw.get("target_id")),
        comment=_as_str(raw.get("comment")),
        likes=raw.get("likes") if isinstance(raw.get("likes"), list) else [],
    )


def _parse_request(raw: dict[str, Any]) -> RequestEvent:
    return RequestEvent(
        raw=raw,
        time=_as_int(raw.get("time")),
        self_id=_as_int(raw.get("self_id")),
        post_type="request",
        request_type=_as_str(raw.get("request_type")),
        sub_type=_as_str(raw.get("sub_type")),
        user_id=_as_int(raw.get("user_id")),
        group_id=_as_int(raw.get("group_id")),
        comment=_as_str(raw.get("comment")),
        flag=_as_str(raw.get("flag")),
    )


def _parse_meta(raw: dict[str, Any]) -> MetaEvent:
    return MetaEvent(
        raw=raw,
        time=_as_int(raw.get("time")),
        self_id=_as_int(raw.get("self_id")),
        post_type="meta_event",
        meta_event_type=_as_str(raw.get("meta_event_type")),
        sub_type=_as_str(raw.get("sub_type")),
        status=raw.get("status") if isinstance(raw.get("status"), dict) else {},
        interval=_as_int(raw.get("interval")),
    )


def _parse_message_field(message: Any, raw_message: Any) -> Message:
    if isinstance(message, list):
        return Message.of(message)
    if isinstance(message, str):
        return Message.of(message)
    if isinstance(message, dict):
        return Message.of([message])
    if raw_message:
        return Message.of(str(raw_message))
    return Message()


def _is_at_me_seg(seg: Segment, self_id: str) -> bool:
    return seg.type == "at" and str(seg.data.get("qq", "")) == self_id


def _check_to_me(event: MessageEvent) -> bool:
    """私聊、开头/结尾 @我、或回复我，才算 to_me。"""
    if event.message_type == "private":
        return True
    self_id = str(event.self_id)
    reply = event.get("reply")
    if isinstance(reply, dict):
        sender = reply.get("sender") or {}
        if isinstance(sender, dict) and str(sender.get("user_id") or "") == self_id:
            return True
    segs = [seg for seg in event.message if not (seg.type == "text" and not str(seg.data.get("text") or "").strip())]
    while segs and segs[0].type == "reply":
        segs = segs[1:]
    if segs and (_is_at_me_seg(segs[0], self_id) or _is_at_me_seg(segs[-1], self_id)):
        return True
    return False


def extract_at_ids(message: Message) -> list[int]:
    ids: list[int] = []
    for seg in message:
        if seg.type != "at":
            continue
        qq = seg.data.get("qq")
        if qq in (None, "all"):
            continue
        try:
            ids.append(int(qq))
        except (TypeError, ValueError):
            continue
    return ids


def first_reply_id(message: Message) -> int | None:
    for seg in message:
        if seg.type == "reply":
            try:
                return int(seg.data.get("id"))
            except (TypeError, ValueError):
                return None
    return None


# 给插件一个顺手的别名
At = Segment.at
Reply = Segment.reply
Image = Segment.image
Text = Segment.text