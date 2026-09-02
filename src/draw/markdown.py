import math
import re
from dataclasses import dataclass, field

from PIL import Image

from .img_utils import concat_images
from .painter import DEFAULT_BOLD_FONT, DEFAULT_FONT, DEFAULT_HEAVY_FONT
from .plot import (
    Canvas,
    FillBg,
    Frame,
    HSplit,
    RoundRectBg,
    Spacer,
    TextBox,
    TextStyle,
    VSplit,
)

PAGE = (255, 248, 242, 255)
INK = (62, 44, 38, 255)
INK2 = (122, 98, 88, 255)
ACCENT = (255, 204, 170, 255)
ACCENT_DEEP = (232, 138, 96, 255)
CARD = (255, 255, 255, 255)
CHIP = (255, 232, 214, 255)
QUOTE = (255, 239, 226, 255)
CODE_BG = (72, 52, 46, 255)
CODE_FG = (255, 236, 220, 255)
RULE = (255, 214, 186, 255)
ADMIN = (196, 86, 64, 255)

_H1 = TextStyle(DEFAULT_HEAVY_FONT, 30, INK)
_H2 = TextStyle(DEFAULT_BOLD_FONT, 24, INK)
_H3 = TextStyle(DEFAULT_BOLD_FONT, 20, INK)
_BODY = TextStyle(DEFAULT_FONT, 16, INK)
_MUTED = TextStyle(DEFAULT_FONT, 15, INK2)
_CHIP = TextStyle(DEFAULT_BOLD_FONT, 15, ACCENT_DEEP)
_CODE = TextStyle(DEFAULT_FONT, 14, CODE_FG)
_FOOT = TextStyle(DEFAULT_FONT, 13, INK2)
_ADMIN = TextStyle(DEFAULT_BOLD_FONT, 14, ADMIN)

_INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")
_CMD_TOKEN_RE = re.compile(r"`([^`]+)`")


@dataclass
class MdBlock:
    kind: str
    text: str = ""
    items: list[str] = field(default_factory=list)


def _strip_md_link(text: str) -> str:
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)


def _is_cmd_line(text: str) -> bool:
    s = text.strip()
    if "`" not in s:
        return False
    leftover = _CMD_TOKEN_RE.sub("", s).replace("🛠️", "").strip()
    return leftover == ""


def _parse_blocks(md: str) -> list[MdBlock]:
    lines = md.replace("\r\n", "\n").split("\n")
    blocks: list[MdBlock] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            body: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            if i < n:
                i += 1
            blocks.append(MdBlock("code", "\n".join(body), items=[lang]))
            continue
        if stripped in ("---", "***", "___"):
            blocks.append(MdBlock("hr"))
            i += 1
            continue
        if stripped.startswith("# "):
            blocks.append(MdBlock("h1", stripped[2:].strip()))
            i += 1
            continue
        if stripped.startswith("## "):
            blocks.append(MdBlock("h2", stripped[3:].strip()))
            i += 1
            continue
        if stripped.startswith("### "):
            blocks.append(MdBlock("h3", stripped[4:].strip()))
            i += 1
            continue
        if stripped.startswith(">"):
            quotes: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                quotes.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            blocks.append(MdBlock("quote", "\n".join(quotes)))
            continue
        if re.match(r"^[-*] ", stripped) or re.match(r"^\d+\.\s", stripped):
            items: list[str] = []
            while i < n:
                cur = lines[i].strip()
                if re.match(r"^[-*] ", cur):
                    items.append(cur[2:].strip())
                elif re.match(r"^\d+\.\s", cur):
                    items.append(re.sub(r"^\d+\.\s+", "", cur))
                else:
                    break
                i += 1
            blocks.append(MdBlock("ul", items=items))
            continue
        para: list[str] = []
        while i < n:
            cur = lines[i]
            s = cur.strip()
            if not s:
                break
            if s.startswith("#") or s.startswith(">") or s.startswith("```") or s in ("---", "***", "___"):
                break
            if re.match(r"^[-*] ", s) or re.match(r"^\d+\.\s", s):
                break
            para.append(s)
            i += 1
        text = " ".join(para)
        if _is_cmd_line(text):
            blocks.append(MdBlock("cmds", text))
        else:
            blocks.append(MdBlock("p", text))
    return blocks


def _inline_flow(text: str, width: int, style: TextStyle = _BODY):
    text = _strip_md_link(text).replace("  ", " ")
    parts = _INLINE_RE.split(text)
    with HSplit().set_sep(0).set_item_align("l").set_padding(0) as row:
        row.set_w(width)
        with VSplit().set_sep(4).set_item_align("l").set_w(width):
            buf = ""
            line_w = width

            def flush_text(chunk: str, chip=False):
                nonlocal buf
                if chip:
                    if buf:
                        TextBox(buf, style, wrap=True, use_real_line_count=True).set_w(line_w)
                        buf = ""
                    TextBox(chunk, _CHIP).set_padding((8, 3)).set_bg(RoundRectBg(CHIP, 8))
                    return
                buf += chunk

            for part in parts:
                if not part:
                    continue
                if part.startswith("**") and part.endswith("**") and len(part) >= 4:
                    if buf:
                        TextBox(buf, style, wrap=True, use_real_line_count=True).set_w(line_w)
                        buf = ""
                    TextBox(part[2:-2], style.replace(font=DEFAULT_BOLD_FONT), wrap=True, use_real_line_count=True).set_w(line_w)
                elif part.startswith("`") and part.endswith("`") and len(part) >= 2:
                    flush_text(part[1:-1], chip=True)
                else:
                    buf += part
            if buf:
                TextBox(buf, style, wrap=True, use_real_line_count=True).set_w(line_w)
    return row


def _cmd_chips(text: str, width: int):
    cmds = _CMD_TOKEN_RE.findall(text)
    admin = "🛠️" in text
    with VSplit().set_sep(8).set_item_align("l").set_w(width) as box:
        if admin:
            TextBox("超级管理", _ADMIN).set_padding((8, 3)).set_bg(RoundRectBg((255, 224, 214, 255), 8))
        with HSplit().set_sep(8).set_item_align("l"):
            for cmd in cmds:
                TextBox(cmd, _CHIP).set_padding((10, 5)).set_bg(RoundRectBg(CHIP, 10))
    return box


def _heading_bar(text: str, style: TextStyle, width: int, bar_h: int):
    with VSplit().set_sep(8).set_item_align("l").set_w(width) as box:
        TextBox(text, style, wrap=True, use_real_line_count=True).set_w(width)
        Spacer(min(180, width // 2), bar_h).set_bg(RoundRectBg(ACCENT_DEEP, 2))
    return box


def _render_blocks(blocks: list[MdBlock], width: int, footer: str | None):
    inner = width - 48
    with Canvas(bg=FillBg(PAGE)).set_padding(0) as canvas:
        with VSplit().set_w(width).set_padding(24).set_sep(16).set_item_align("l"):
            Spacer(width - 48, 6).set_bg(RoundRectBg(ACCENT, 4))
            for block in blocks:
                if block.kind == "h1":
                    _heading_bar(block.text, _H1, inner, 5)
                elif block.kind == "h2":
                    _heading_bar(block.text, _H2, inner, 4)
                elif block.kind == "h3":
                    with HSplit().set_sep(10).set_item_align("l").set_w(inner):
                        Spacer(6, 28).set_bg(RoundRectBg(ACCENT_DEEP, 3))
                        TextBox(block.text, _H3, wrap=True, use_real_line_count=True).set_w(inner - 16)
                elif block.kind == "hr":
                    Spacer(inner, 2).set_bg(FillBg(RULE))
                elif block.kind == "cmds":
                    _cmd_chips(block.text, inner)
                elif block.kind == "quote":
                    with HSplit().set_sep(0).set_item_align("t").set_w(inner).set_bg(RoundRectBg(QUOTE, 12)):
                        Spacer(8, 8).set_bg(FillBg(ACCENT_DEEP))
                        with VSplit().set_padding((14, 12)).set_sep(6).set_item_align("l").set_w(inner - 8):
                            for line in block.text.split("\n"):
                                if line.strip():
                                    _inline_flow(line, inner - 36, _MUTED)
                elif block.kind == "ul":
                    with VSplit().set_sep(8).set_item_align("l").set_w(inner):
                        for item in block.items:
                            with HSplit().set_sep(8).set_item_align("t").set_w(inner):
                                TextBox("●", TextStyle(DEFAULT_BOLD_FONT, 14, ACCENT_DEEP)).set_padding((0, 2))
                                _inline_flow(item, inner - 28)
                elif block.kind == "code":
                    with Frame().set_bg(RoundRectBg(CODE_BG, 12)).set_padding(14).set_w(inner):
                        TextBox(block.text or " ", _CODE, wrap=True, use_real_line_count=True).set_w(inner - 28)
                else:
                    _inline_flow(block.text, inner)
            if footer:
                Spacer(inner, 2).set_bg(FillBg(RULE))
                TextBox(footer, _FOOT, wrap=True, use_real_line_count=True).set_w(inner)
    return canvas


def _split_long_image(image: Image.Image, width: int, intersect: int) -> Image.Image:
    max_height = width * 3
    if image.height <= max_height:
        return image
    n = max(2, math.floor(math.sqrt(image.height * image.width) / image.width))
    height = math.ceil(image.height / n)
    pieces = []
    for y in range(0, image.height, height):
        top = max(0, y - intersect) if y else 0
        pieces.append(image.crop((0, top, image.width, min(image.height, y + height))))
    return concat_images(pieces, "h")


async def markdown_to_image(
    markdown_text: str,
    width: int = 640,
    scale: float = 1.0,
    intersect: int = 24,
    footer: str | None = None,
) -> Image.Image:
    blocks = _parse_blocks(markdown_text)
    if not blocks:
        blocks = [MdBlock("p", "（空帮助）")]
    canvas = _render_blocks(blocks, width, footer)
    image = await canvas.get_img(scale=scale if scale and scale != 1 else None)
    return _split_long_image(image, image.width, intersect)
