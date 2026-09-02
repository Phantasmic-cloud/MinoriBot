from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Union

MessageLike = Union["Message", "Segment", str, list, dict, None]


def escape_text(s: str) -> str:
    return s.replace("&", "&amp;").replace("[", "&#91;").replace("]", "&#93;")


def escape_cq(s: str) -> str:
    return escape_text(s).replace(",", "&#44;")


def unescape_text(s: str) -> str:
    return s.replace("&#91;", "[").replace("&#93;", "]").replace("&amp;", "&")


def unescape_cq(s: str) -> str:
    return unescape_text(s.replace("&#44;", ","))


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


@dataclass
class Segment:
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def text(cls, text: str) -> Segment:
        return cls("text", {"text": str(text)})

    @classmethod
    def at(cls, qq: int | str) -> Segment:
        return cls("at", {"qq": str(qq)})

    @classmethod
    def reply(cls, message_id: int | str) -> Segment:
        return cls("reply", {"id": str(message_id)})

    @classmethod
    def image(cls, file: str) -> Segment:
        return cls("image", {"file": file})

    @classmethod
    def record(cls, file: str) -> Segment:
        return cls("record", {"file": file})

    @classmethod
    def face(cls, face_id: int | str) -> Segment:
        return cls("face", {"id": str(face_id)})

    def to_cq(self) -> str:
        if self.type == "text":
            return escape_text(_stringify(self.data.get("text", "")))
        body = ",".join(f"{k}={escape_cq(_stringify(v))}" for k, v in self.data.items())
        if body:
            return f"[CQ:{self.type},{body}]"
        return f"[CQ:{self.type}]"

    def to_rich(self) -> dict[str, Any]:
        return {"type": self.type, "data": {k: _stringify(v) for k, v in self.data.items()}}

    def __str__(self) -> str:
        if self.type == "text":
            return _stringify(self.data.get("text", ""))
        return self.to_cq()

    def __add__(self, other: MessageLike) -> Message:
        return Message.of(self) + Message.of(other)

    def __radd__(self, other: MessageLike) -> Message:
        return Message.of(other) + Message.of(self)


class Message(list):
    """消息段列表。可从 CQ 字符串、segment 数组或混用输入构造。"""

    def __init__(self, segs: Iterable[Segment] | None = None) -> None:
        super().__init__(list(segs or []))

    @classmethod
    def of(cls, obj: MessageLike) -> Message:
        if obj is None:
            return cls()
        if isinstance(obj, Message):
            return cls(obj)
        if isinstance(obj, Segment):
            return cls([obj])
        if isinstance(obj, str):
            return parse_cq(obj)
        if isinstance(obj, dict):
            return cls([_seg_from_dict(obj)])
        if isinstance(obj, list):
            segs: list[Segment] = []
            for item in obj:
                if isinstance(item, Segment):
                    segs.append(item)
                elif isinstance(item, str):
                    segs.extend(parse_cq(item))
                elif isinstance(item, dict):
                    segs.append(_seg_from_dict(item))
                elif isinstance(item, Message):
                    segs.extend(item)
                else:
                    segs.append(Segment.text(str(item)))
            return cls(segs)
        return cls([Segment.text(str(obj))])

    @property
    def plain_text(self) -> str:
        return "".join(_stringify(s.data.get("text", "")) for s in self if s.type == "text")

    def to_rich(self) -> list[dict[str, Any]]:
        return [s.to_rich() for s in self]

    def to_cq(self) -> str:
        return "".join(s.to_cq() for s in self)

    def __str__(self) -> str:
        return self.to_cq()

    def __add__(self, other: MessageLike) -> Message:
        return Message(list(self) + list(Message.of(other)))

    def __radd__(self, other: MessageLike) -> Message:
        return Message.of(other) + self

    def __iadd__(self, other: MessageLike) -> Message:
        self.extend(Message.of(other))
        return self

    def __iter__(self) -> Iterator[Segment]:
        return super().__iter__()


def _seg_from_dict(item: dict[str, Any]) -> Segment:
    return Segment(str(item.get("type") or "text"), dict(item.get("data") or {}))


def parse_cq(s: str) -> Message:
    if not s:
        return Message()
    segs: list[Segment] = []
    buf: list[str] = []
    i, n = 0, len(s)
    while i < n:
        if s.startswith("[CQ:", i):
            end = s.find("]", i)
            if end == -1:
                buf.append(s[i])
                i += 1
                continue
            if buf:
                segs.append(Segment.text(unescape_text("".join(buf))))
                buf.clear()
            inner = s[i + 4 : end]
            if "," in inner:
                typ, rest = inner.split(",", 1)
                data: dict[str, Any] = {}
                for part in rest.split(","):
                    if not part:
                        continue
                    if "=" in part:
                        k, v = part.split("=", 1)
                        data[k] = unescape_cq(v)
                    else:
                        data[part] = ""
                segs.append(Segment(typ, data))
            else:
                segs.append(Segment(inner, {}))
            i = end + 1
        else:
            buf.append(s[i])
            i += 1
    if buf:
        segs.append(Segment.text(unescape_text("".join(buf))))
    return Message(segs)


def dump_message(message: MessageLike, auto_escape: bool = False) -> str | list[dict[str, Any]]:
    """转成 OneBot 发送用的 message 字段。默认走 array 格式。"""
    if auto_escape:
        if isinstance(message, str):
            return message
        return Message.of(message).plain_text
    return Message.of(message).to_rich()
