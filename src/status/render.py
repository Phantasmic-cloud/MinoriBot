import random
from pathlib import Path

from PIL import Image, ImageDraw

from src.utils import *

from .collectors import StatusData
from .util import auto_convert_byte, percent_color

CARD_W = 618
CANVAS_W = 650
AVATAR_SIZE = 125
DONUT_SIZE = 150
DONUT_STROKE = 15

TEXT = (58, 58, 58, 255)
TEXT2 = (106, 106, 106, 255)
CARD_BG = (250, 250, 250, 170)
MASK = (250, 250, 250, 102)
GRAY_BG = (190, 190, 190, 170)
PURPLE = (154, 119, 207, 170)
GREEN = (29, 169, 18, 170)
BLUE = (17, 141, 195, 170)
ORANGE = (238, 144, 37, 170)
GRAY_LABEL = (190, 190, 190, 170)
SHADOW = (0, 0, 0, 40)

BG_DIR = Path("data/status/bg")
DEFAULT_AVATAR = Path("data/status/default_avatar.webp")
config = Config("status")

_title = TextStyle(font=DEFAULT_BOLD_FONT, size=32, color=TEXT)
_nick = TextStyle(font=DEFAULT_BOLD_FONT, size=36, color=TEXT)
_body = TextStyle(font=DEFAULT_FONT, size=20, color=TEXT)
_small = TextStyle(font=DEFAULT_FONT, size=16, color=TEXT2)
_tiny = TextStyle(font=DEFAULT_FONT, size=12, color=TEXT2)
_footer = TextStyle(font=DEFAULT_FONT, size=14, color=TEXT)
_label = TextStyle(font=DEFAULT_FONT, size=16, color=TEXT2)
_donut_num = TextStyle(font=DEFAULT_BOLD_FONT, size=32, color=TEXT)


def _card_bg() -> RoundRectBg:
    return RoundRectBg(CARD_BG, 8, blurglass=True, blurglass_kwargs={"blur": 2, "shadow_width": 4, "shadow_alpha": 0.18})


def _label_box(text: str, fill) -> TextBox:
    return (
        TextBox(text, _label)
        .set_padding((6, 3))
        .set_bg(RoundRectBg(fill, 4))
        .set_margin((0, 2))
    )


def _circle_avatar(img: Image.Image | None, size: int = AVATAR_SIZE) -> Image.Image:
    if img is None:
        if DEFAULT_AVATAR.exists():
            img = Image.open(DEFAULT_AVATAR).convert("RGBA")
        else:
            img = Image.new("RGBA", (size, size), (200, 200, 200, 255))
    img = img.convert("RGBA").resize((size, size), Image.Resampling.BILINEAR)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((1, 1, size - 2, size - 2), fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0))
    out.putalpha(mask)
    return out


class Donut(Frame):
    def __init__(self, percent: float | None, title: str, caption: str):
        super().__init__()
        self.percent = percent
        self.title = title
        self.caption = caption
        self.set_size((DONUT_SIZE + 8, None))
        self.set_content_align("t")
        with self:
            with VSplit().set_sep(4).set_item_align("c"):
                Frame().set_size((DONUT_SIZE, DONUT_SIZE)).add_draw_func(self._draw_ring)
                TextBox(title, _title).set_padding(0)
                TextBox(caption, _tiny, wrap=True, use_real_line_count=True).set_w(DONUT_SIZE).set_content_align("c")

    def _draw_ring(self, _w, p):
        size = (DONUT_SIZE, DONUT_SIZE)
        p.pieslice((0, 0), size, 0, 360, GRAY_BG)
        if self.percent is not None:
            sweep = max(0.0, min(100.0, float(self.percent))) * 3.6
            if sweep > 0:
                p.pieslice((0, 0), size, -90, -90 + sweep, percent_color(self.percent))
        inner = DONUT_SIZE - DONUT_STROKE * 2
        p.roundrect(
            (DONUT_STROKE, DONUT_STROKE),
            (inner, inner),
            (250, 250, 250, 230),
            inner // 2,
        )
        label = "未部署" if self.percent is None else f"{self.percent:.0f}%"
        tb = TextBox(label, _donut_num)
        tw, th = tb._get_self_size()
        p.move_region(((DONUT_SIZE - tw) // 2, (DONUT_SIZE - th) // 2), (tw, th))
        tb.draw(p)
        p.restore_region()


class ProgressBar(Frame):
    def __init__(self, percent: float | None, text: str, w: int = 280, h: int = 22):
        super().__init__()
        self.percent = percent
        self.text = text
        self.set_size((w, h))
        self.add_draw_func(self._draw_bar)
        with self:
            TextBox(text, _small, overflow="shrink").set_w(max(1, w - 8)).set_content_align("c")

    def _draw_bar(self, _w, p):
        p.roundrect((0, 0), p.size, GRAY_BG, 4)
        if self.percent is not None:
            pw = int(p.w * max(0.0, min(100.0, float(self.percent))) / 100)
            if pw > 0:
                p.roundrect((0, 0), (pw, p.h), percent_color(self.percent), 4, corners=(True, False, False, True) if pw < p.w else (True, True, True, True))


def _pick_bg() -> Image.Image | None:
    if not BG_DIR.exists():
        return None
    files = [p for p in BG_DIR.iterdir() if p.is_file()]
    if not files:
        return None
    try:
        return Image.open(random.choice(files)).convert("RGBA")
    except Exception:
        return None


def _header(data: StatusData):
    with VSplit().set_w(CARD_W).set_padding(16).set_sep(12).set_item_align("l").set_bg(_card_bg()) as card:
        for bot in data.bots:
            with HSplit().set_sep(16).set_item_align("l"):
                ImageBox(_circle_avatar(bot.avatar), size=(AVATAR_SIZE, AVATAR_SIZE))
                with VSplit().set_sep(8).set_item_align("l").set_w(CARD_W - 32 - AVATAR_SIZE - 16):
                    TextBox(bot.nick, _nick, wrap=True, use_real_line_count=True)
                    with Flow(hsep=6, vsep=4).set_w(CARD_W - 32 - AVATAR_SIZE - 16):
                        _label_box(bot.adapter, PURPLE)
                        _label_box(f"Bot已连接 {bot.bot_connected}", GREEN)
                        _label_box(f"收 {bot.msg_rec}", BLUE)
                        _label_box(f"发 {bot.msg_sent}", ORANGE)
        with Flow(hsep=8, vsep=4).set_w(CARD_W - 32):
            _label_box(f"MinoriBot运行 {data.bot_run_time}", GRAY_LABEL)
            _label_box(f"系统运行 {data.system_run_time}", GRAY_LABEL)
    return card


def _cpu_mem(data: StatusData):
    cm = data.cpu_mem
    if cm is None:
        return Spacer(1, 1)
    cores = cm.cpu_count if cm.cpu_count is not None else "??"
    logical = cm.cpu_count_logical if cm.cpu_count_logical is not None else "??"
    ram = f"{auto_convert_byte(cm.ram_used)} / {auto_convert_byte(cm.ram_total)}"
    if cm.swap_percent is None:
        swap_caption = "未部署"
    else:
        swap_caption = f"{auto_convert_byte(cm.swap_used)} / {auto_convert_byte(cm.swap_total)}"
    with HSplit().set_w(CARD_W).set_padding(16).set_sep(8).set_item_align("t").set_item_size_mode("expand").set_bg(_card_bg()) as card:
        Donut(cm.cpu_percent, "CPU", f"{cores}核 {logical}线程 {cm.cpu_freq}\n{cm.cpu_brand}")
        Donut(cm.ram_percent, "RAM", ram)
        Donut(cm.swap_percent, "SWAP", swap_caption)
    return card


def _disk(data: StatusData):
    if not data.disk_usage and not data.disk_io:
        return Spacer(1, 1)
    inner_w = CARD_W - 32
    bar_w = inner_w - 150 - 70 - 16
    with VSplit().set_w(CARD_W).set_padding(16).set_sep(10).set_item_align("l").set_bg(_card_bg()) as card:
        for it in data.disk_usage:
            with HSplit().set_sep(8).set_item_align("c").set_w(inner_w):
                TextBox(it.name, _body, overflow="shrink").set_w(150)
                if it.exception:
                    ProgressBar(None, it.exception, w=bar_w)
                    TextBox("??.?%", _body).set_w(70).set_content_align("r")
                else:
                    ProgressBar(it.percent, f"{auto_convert_byte(it.used)} / {auto_convert_byte(it.total)}", w=bar_w)
                    TextBox(f"{it.percent:.1f}%" if it.percent is not None else "??.?%", _body).set_w(70).set_content_align("r")
        if data.disk_io:
            for it in data.disk_io:
                with HSplit().set_sep(8).set_item_align("c").set_w(inner_w):
                    TextBox(it.name, _body, overflow="shrink")
                    FlexSpacer()
                    TextBox("读", _small)
                    TextBox(auto_convert_byte(it.read, suffix="/s"), _body).set_content_align("r")
                    TextBox("|", _small)
                    TextBox("写", _small)
                    TextBox(auto_convert_byte(it.write, suffix="/s"), _body).set_content_align("r")
    return card


def _network(data: StatusData):
    if not data.network_io and not data.network_connection:
        return Spacer(1, 1)
    inner_w = CARD_W - 32
    with VSplit().set_w(CARD_W).set_padding(16).set_sep(8).set_item_align("l").set_bg(_card_bg()) as card:
        for it in data.network_io:
            with HSplit().set_sep(8).set_item_align("c").set_w(inner_w):
                TextBox(it.name, _body, overflow="shrink")
                FlexSpacer()
                TextBox("↑", _small)
                TextBox(auto_convert_byte(it.sent, suffix="/s"), _body).set_content_align("r")
                TextBox("|", _small)
                TextBox("↓", _small)
                TextBox(auto_convert_byte(it.recv, suffix="/s"), _body).set_content_align("r")
        for it in data.network_connection:
            with HSplit().set_sep(8).set_item_align("c").set_w(inner_w):
                TextBox(it.name, _body, overflow="shrink")
                FlexSpacer()
                if it.error:
                    TextBox(it.error, _body).set_content_align("r")
                else:
                    TextBox(f"{it.status} {it.reason}", _body)
                    TextBox("|", _small)
                    TextBox(f"{it.delay:.2f}ms" if it.delay is not None else "-", _body).set_content_align("r")
    return card


def _process(data: StatusData):
    if not data.process_status:
        return Spacer(1, 1)
    inner_w = CARD_W - 32
    with VSplit().set_w(CARD_W).set_padding(16).set_sep(6).set_item_align("l").set_bg(_card_bg()) as card:
        for it in data.process_status:
            with HSplit().set_sep(8).set_item_align("c").set_w(inner_w):
                TextBox(it.name, _body, overflow="shrink")
                FlexSpacer()
                TextBox("CPU", _small)
                TextBox(f"{it.cpu:.1f}%", _body).set_content_align("r")
                TextBox("|", _small)
                TextBox("MEM", _small)
                TextBox(auto_convert_byte(it.mem), _body).set_content_align("r")
    return card


async def render_status(data: StatusData) -> Image.Image:
    bg_img = _pick_bg()
    if bg_img is not None:
        canvas_bg = ImageBg(bg_img, mode="fit", blur=False, fade=0.08)
    else:
        canvas_bg = FillBg((58, 58, 58, 255))

    with Canvas(w=CANVAS_W, bg=canvas_bg).set_padding(0) as canvas:
        with VSplit().set_w(CANVAS_W).set_padding(16).set_sep(16).set_item_align("c").set_bg(FillBg(MASK)):
            _header(data)
            _cpu_mem(data)
            if data.disk_usage or data.disk_io:
                _disk(data)
            if data.network_io or data.network_connection:
                _network(data)
            if data.process_status:
                _process(data)
            TextBox(
                f"MinoriBot × PicStatus | {data.time}\n{data.python_version} | {data.system_name}",
                _footer,
                wrap=True,
                use_real_line_count=True,
            ).set_w(CARD_W).set_content_align("c")

    scale = float(config.get("scale", 2) or 1)
    return await canvas.get_img(scale=scale)