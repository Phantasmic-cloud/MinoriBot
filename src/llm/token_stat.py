# Token 用量统计：每次 LLM / embedding / TTS 调用记一行，/token stat 查询
import asyncio
import math
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from src.utils import *

logger = get_logger("llm")
file_db = get_file_db("data/llm/db.json")
cd = ColdDown(file_db, logger, cold_down_name="token_stat")
stat_config = Config("llm.stat")

DB_PATH = "data/llm/token_usage.sqlite"

_conn: aiosqlite.Connection | None = None
_write_lock = asyncio.Lock()


async def _get_conn() -> aiosqlite.Connection:
    global _conn
    if _conn is None:
        create_parent_folder(DB_PATH)
        _conn = await aiosqlite.connect(DB_PATH)
        # 旧版本表带 source 列，与新结构不符时重建
        cursor = await _conn.execute("PRAGMA table_info(token_usage)")
        cols = [row[1] for row in await cursor.fetchall()]
        await cursor.close()
        if cols and "source" in cols:
            await _conn.execute("DROP TABLE token_usage")
            logger.info("token_usage 表结构已变更，重建统计表")
        await _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time REAL NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                kind TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        await _conn.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_time ON token_usage(time)")
        await _conn.commit()
        logger.info("连接token统计数据库 %s 成功", DB_PATH)
    return _conn


async def record_token_usage(
    provider: str,
    model: str,
    kind: str,
    prompt_tokens: int,
    completion_tokens: int = 0,
):
    """记录一次调用的 token 用量。任何失败只 warning，不影响主流程。

    kind: chat / embedding / tts（tts 时 prompt_tokens 为字符数）
    """
    try:
        async with _write_lock:
            conn = await _get_conn()
            await conn.execute(
                "INSERT INTO token_usage (time, provider, model, kind, prompt_tokens, completion_tokens)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    str(provider),
                    str(model),
                    str(kind),
                    int(prompt_tokens or 0),
                    int(completion_tokens or 0),
                ),
            )
            await conn.commit()
    except Exception as e:
        logger.warning("记录token用量失败: %s", get_exc_desc(e))


# ======================= 查询 ======================= #


async def query_token_usage(start: datetime, end: datetime) -> dict:
    """聚合 [start, end) 范围内的用量。

    返回：
      total      —— 全部调用的合计（prompt/completion/calls）
      chat       —— 只含 chat：{total, by_model(带输入输出), series(按小时时序)}
      extras     —— embedding / tts 明细列表（每条含 kind）
    """
    conn = await _get_conn()
    cursor = await conn.execute(
        """
        SELECT provider, model, kind,
               SUM(prompt_tokens), SUM(completion_tokens), COUNT(*)
        FROM token_usage
        WHERE time >= ? AND time < ?
        GROUP BY provider, model, kind
        """,
        (start.timestamp(), end.timestamp()),
    )
    rows = await cursor.fetchall()
    await cursor.close()

    total = {"prompt": 0, "completion": 0, "calls": 0}
    chat_total = {"prompt": 0, "completion": 0, "calls": 0}
    chat_models: dict[tuple, dict] = {}
    extras: list[dict] = []

    for provider, model, kind, prompt, completion, calls in rows:
        prompt, completion, calls = int(prompt or 0), int(completion or 0), int(calls or 0)
        provider, model, kind = str(provider), str(model), str(kind)
        total["prompt"] += prompt
        total["completion"] += completion
        total["calls"] += calls

        if kind == "chat":
            chat_total["prompt"] += prompt
            chat_total["completion"] += completion
            chat_total["calls"] += calls
            row = chat_models.setdefault(
                (provider, model),
                {"label": f"{provider}:{model}", "prompt": 0, "completion": 0, "calls": 0},
            )
            row["prompt"] += prompt
            row["completion"] += completion
            row["calls"] += calls
        else:
            extras.append({
                "label": f"{provider}:{model}",
                "kind": kind,
                "prompt": prompt,
                "completion": completion,
                "calls": calls,
            })

    chat_models_list = sorted(
        chat_models.values(), key=lambda r: (r["prompt"] + r["completion"], r["calls"]), reverse=True
    )
    extras.sort(key=lambda r: (r["prompt"] + r["completion"], r["calls"]), reverse=True)

    # 按小时时序（只统计 chat）
    cursor = await conn.execute(
        """
        SELECT time, prompt_tokens, completion_tokens
        FROM token_usage
        WHERE time >= ? AND time < ? AND kind = 'chat'
        """,
        (start.timestamp(), end.timestamp()),
    )
    time_rows = await cursor.fetchall()
    await cursor.close()

    buckets = _make_time_buckets(start, end)
    for ts, prompt, completion in time_rows:
        idx = int((float(ts) - start.timestamp()) // buckets["step"])
        if 0 <= idx < len(buckets["labels"]):
            b = buckets["buckets"][idx]
            b["tokens"] += int(prompt or 0) + int(completion or 0)
            b["calls"] += 1

    return {
        "total": total,
        "chat": {"total": chat_total, "by_model": chat_models_list, "series": buckets},
        "extras": extras,
    }


def _make_time_buckets(start: datetime, end: datetime) -> dict:
    """按范围长度自适应分桶：先按目标桶数选粒度，再用该粒度铺满区间。

    粒度候选（秒）：1/5/10/15/30 分钟、1 小时、1 天、1 周。
    短区间（如 1h）会切到分钟级，长区间退到按天/按周。
    """
    span = (end - start).total_seconds()
    target_buckets = 31
    candidates = (
        (60.0, "m"), (120.0, "m"), (300.0, "m"), (600.0, "m"), (900.0, "m"), (1800.0, "m"),
        (3600.0, "h"),
        (86400.0, "d"),
        (7 * 86400.0, "w"),
        (30 * 86400.0, "mo"),
    )
    step, unit = candidates[-1]
    for cand_step, cand_unit in candidates:
        if span / cand_step <= target_buckets:
            step, unit = cand_step, cand_unit
            break

    def label_of(t: datetime) -> str:
        if unit == "m":
            return f"{t:%H:%M}"
        if unit == "h":
            return f"{t:%H:%M}" if t.hour else f"{t.month}月{t.day}日0时"
        if unit == "mo":
            return f"{t.year}/{t.month}"
        return f"{t.month}/{t.day}"

    labels = []
    cur = start
    step_delta = timedelta(seconds=step)
    while cur < end:
        labels.append(label_of(cur))
        cur += step_delta
    if not labels:
        labels = [label_of(start)]
    return {
        "step": step,
        "labels": labels,
        "buckets": [{"tokens": 0, "calls": 0} for _ in labels],
    }


# ======================= 时间解析 ======================= #

_RANGE_HELP = (
    "时间参数格式:\n"
    "  空                      过去 24 小时\n"
    "  today                   今天 00:00 起\n"
    "  30m / 1h / 24h / 7d     过去 N 分钟/小时/天\n"
    "  2026-10-13              指定日期\n"
    "  2026-10-10~2026-10-13   日期区间"
)


def parse_token_range(args: str) -> tuple[datetime, datetime]:
    """把命令参数解析成 [start, end) 时间范围。"""
    args = (args or "").strip()
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if not args:
        return now - timedelta(hours=24), now

    if args.lower() == "today":
        return today, today + timedelta(days=1)

    m = re.fullmatch(r"(\d+)\s*([mhd])", args, re.IGNORECASE)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        delta = {"m": timedelta(minutes=n), "h": timedelta(hours=n), "d": timedelta(days=n)}[unit]
        return now - delta, now

    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*[~～]\s*(\d{4}-\d{2}-\d{2})", args)
    if m:
        start = datetime.strptime(m.group(1), "%Y-%m-%d")
        end = datetime.strptime(m.group(2), "%Y-%m-%d") + timedelta(days=1)
        if end <= start:
            raise ValueError("结束日期必须晚于开始日期")
        return start, end

    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})", args)
    if m:
        start = datetime.strptime(m.group(1), "%Y-%m-%d")
        return start, start + timedelta(days=1)

    raise ValueError(f"时间参数格式错误\n{_RANGE_HELP}")


def _fmt_range(start: datetime, end: datetime) -> str:
    """紧凑显示：同年省略年份，避免跨天时字符串过长撑爆卡片。"""
    last = end - timedelta(seconds=1)
    d = "%m-%d" if start.year == last.year else "%Y-%m-%d"
    if start.hour == 0 and start.minute == 0 and last.hour == 23 and last.minute == 59:
        if start.date() == last.date():
            return f"{start:%Y-%m-%d} 全天"
        return f"{start.strftime(d)} ~ {last.strftime(d)}"
    if start.date() == last.date():
        return f"{start.strftime(d)} {start:%H:%M} ~ {last:%H:%M}"
    return f"{start.strftime(d)} {start:%H:%M} ~ {last.strftime(d)} {last:%H:%M}"


# ======================= 绘图 ======================= #
# 视觉风格对齐 src/status/render.py：深色渐变底 + 半透遮罩 + 毛玻璃卡片 + 胶囊标签
# 四个卡片：汇总头卡 / 趋势折线 / 模型分布（嵌套圆环+明细）/ 嵌入与语音

CARD_W = 618
CANVAS_W = 650

TEXT = (32, 32, 32, 255)
TEXT2 = (72, 72, 72, 255)
CARD_BG = (250, 250, 250, 170)
MASK = (250, 250, 250, 34)
GRAY_BG = (190, 190, 190, 170)
SEGMENT = (190, 190, 190, 255)
PURPLE = (154, 119, 207, 170)
GREEN = (29, 169, 18, 170)
BLUE = (17, 141, 195, 170)
ORANGE = (238, 144, 37, 170)

_BG_TOP = (62, 70, 105, 255)
_BG_BOTTOM = (20, 21, 32, 255)

# 氛围光斑：为玻璃卡片提供可透的纹理层次，避免背景过平
_DEFAULT_BG_DIR = "data/llm/stat_bg"
_BLUR_COLORS = [
    (96, 116, 190),
    (170, 96, 180),
    (72, 150, 170),
    (196, 126, 86),
]

IN_COLOR = (66, 133, 244, 255)      # #4285F4 输入
OUT_COLOR = (245, 158, 11, 255)     # #F59E0B 输出
OTHER_COLOR = (140, 144, 152, 255)  # 「其它模型」统一灰

# 模型配色，按消耗排名随机取用且不重复
_MODEL_COLORS = [
    (76, 175, 80, 255),
    (233, 93, 154, 255),
    (245, 189, 79, 255),
    (120, 144, 220, 255),
    (63, 191, 191, 255),
    (186, 120, 220, 255),
]

_CHART_GRID = (0, 0, 0, 30)
_CHART_TEXT = (108, 114, 126, 255)

_FONT = "SourceHanSansCN-Regular"
_BOLD = "SourceHanSansCN-Bold"

_title = TextStyle(font=_BOLD, size=26, color=TEXT)
_section = TextStyle(font=_BOLD, size=19, color=TEXT)
_body = TextStyle(font=_FONT, size=17, color=TEXT)
_num = TextStyle(font=_FONT, size=17, color=TEXT2)
_small = TextStyle(font=_FONT, size=16, color=TEXT2)
_dim = TextStyle(font=_FONT, size=16, color=(140, 146, 158, 255))
_label = TextStyle(font=_FONT, size=15, color=TEXT2)
_big = TextStyle(font=_BOLD, size=28, color=TEXT)
_donut_num = TextStyle(font=_BOLD, size=22, color=TEXT)

# 明细行：名字列吃剩余宽度，右侧指标由 Grid 自适应列宽（参照 status._metric_table）
_ROW_SEP = 8
_IOBAR_W = 78

_CHART_H = 210
_DONUT_SIZE = 168

# 右下角水印（硬编码，不走配置）
_WATERMARK = "Generated by MinoriBot"
_WATERMARK_SIZE = 22


def _pick_bg_image() -> Image.Image | None:
    """从配置的 bg_dir 里随机取一张背景图；目录不存在/为空就返回 None。"""
    try:
        bg_dir = Path(str(stat_config.get("bg_dir", "") or _DEFAULT_BG_DIR))
    except Exception:
        bg_dir = Path(_DEFAULT_BG_DIR)
    if not bg_dir.exists():
        return None
    files = [p for p in bg_dir.iterdir() if p.is_file()]
    if not files:
        return None
    try:
        return Image.open(random.choice(files)).convert("RGBA")
    except Exception:
        return None


def _render_ambient_bg(w: int, h: int) -> Image.Image:
    """程序化氛围底：竖向渐变 + 随机光斑 + 极轻噪点。"""
    img = Image.new("RGB", (w, h), _BG_BOTTOM[:3])
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        d.line(
            (0, y, w, y),
            fill=tuple(int(_BG_TOP[i] + (_BG_BOTTOM[i] - _BG_TOP[i]) * t) for i in range(3)),
        )
    # 光斑：大范围柔和色块，让上方更亮
    glow = Image.new("RGB", (w, h), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    spots = random.randint(3, 5)
    for _ in range(spots):
        cx = random.randint(-w // 4, w + w // 4)
        cy = random.randint(-h // 6, int(h * 0.75))
        rx = random.randint(w // 4, int(w * 0.6))
        ry = random.randint(h // 12, int(h * 0.3))
        color = random.choice(_BLUR_COLORS)
        gd.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=max(24, w // 12)))
    img = ImageChops.screen(img, glow)
    return img


def _noise_layer(w: int, h: int, strength: int = 6) -> Image.Image:
    """细噪点，压掉渐变的塑料感。用 PIL 内置噪声，避免逐像素生成。"""
    small_w = max(2, w // 3)
    small_h = max(2, h // 3)
    noise = Image.effect_noise((small_w, small_h), 40).resize((w, h), Image.Resampling.BILINEAR)
    alpha = Image.new("L", (w, h), strength)
    return Image.merge("RGBA", (noise, noise, noise, alpha))


class _AmbientBg(WidgetBg):
    """氛围背景：有背景图就用图（压暗一档），否则程序生成渐变+光斑+噪点。"""

    def __init__(self, fade: float = None):
        if fade is None:
            try:
                fade = float(stat_config.get("bg_fade", 0.12) or 0.12)
            except (TypeError, ValueError):
                fade = 0.12
        self.fade = max(0.0, min(0.9, fade))
        self._bg = None

    def _ensure(self, size):
        if self._bg is not None:
            return
        w, h = size
        img = _pick_bg_image()
        if img is not None:
            scale = max(w / img.size[0], h / img.size[1])
            iw, ih = max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))
            img = img.resize((iw, ih), Image.Resampling.LANCZOS)
            x = (iw - w) // 2
            y = (ih - h) // 2
            img = img.crop((x, y, x + w, y + h))
            base = img.convert("RGB")
            # 压暗一档，避免底图过亮抢卡片
            base = Image.eval(base, lambda v: int(v * (1 - self.fade)))
        else:
            base = _render_ambient_bg(w, h)
        base = base.convert("RGBA")
        self._bg = Image.alpha_composite(base, _noise_layer(w, h))

    def draw(self, p: Painter):
        self._ensure((p.w, p.h))
        p.paste(self._bg, (0, 0), exclude_on_hash=True)


def _canvas_bg() -> _AmbientBg:
    return _AmbientBg()


def _card_bg() -> RoundRectBg:
    """毛玻璃卡片：blur 让背后的光斑透出来，较大 radius + 阴影撑出层次。"""
    return RoundRectBg(
        CARD_BG, 12, blurglass=True,
        blurglass_kwargs={"blur": 4, "shadow_width": 9, "shadow_alpha": 0.28},
    )


def _chip_bg(fill) -> RoundRectBg:
    return RoundRectBg(
        fill, 5, blurglass=True,
        blurglass_kwargs={"blur": 0, "shadow_width": 3, "shadow_alpha": 0.2},
    )


def _label_box(text: str, fill, w: int = None) -> TextBox:
    box = TextBox(text, _label).set_padding((6, 4)).set_bg(_chip_bg(fill))
    if w:
        box.set_w(w).set_content_align("c")
    return box


def _divider(w: int) -> Spacer:
    return Spacer(w, 2).set_bg(FillBg(SEGMENT))


def _fmt_count(n) -> str:
    return f"{int(n or 0):,}"


def _fmt_token(n) -> str:
    """Token 数紧凑显示：57.8k / 1.23M，避免撑爆数值列。"""
    v = float(n or 0)
    if v < 1e3:
        return f"{v:.0f}"
    if v < 1e6:
        s = f"{v / 1e3:.1f}k" if v < 1e5 else f"{v / 1e3:.0f}k"
        return s.replace(".0k", "k")
    return f"{v / 1e6:.2f}M".replace(".00M", "M")


def _nice_ceil(v: float) -> float:
    """把最大值向上取到 1/2/2.5/5/10 × 10^n，作为坐标轴顶端。"""
    v = float(v or 0)
    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    base = 10.0 ** exp
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * base:
            return m * base
    return 10 * base


def _round_rect(p: Painter, pos, size, fill, radius: int, corners=(True, True, True, True)):
    """Painter.roundrect 的保险封装：radius 超过半高时收敛，避免退化成尖角。"""
    if size[0] <= 0 or size[1] <= 0:
        return
    r = max(0, min(int(radius), size[0] // 2, size[1] // 2))
    p.roundrect(pos, size, fill, r, corners=corners)


class _IORatioBar(Frame):
    """输入 / 输出比例长条：左蓝右橙。"""

    def __init__(self, in_v, out_v, w: int = _IOBAR_W, h: int = 12):
        super().__init__()
        self.in_v = max(0.0, float(in_v or 0))
        self.out_v = max(0.0, float(out_v or 0))
        self.set_size((w, h))
        self.add_draw_func(self._draw)

    def _draw(self, _widget, p: Painter):
        total = self.in_v + self.out_v
        if total <= 0:
            return
        r = p.h // 2
        _round_rect(p, (0, 0), (p.w, p.h), OUT_COLOR, r)
        in_w = min(p.w, max(2, int(round(p.w * self.in_v / total))))
        _round_rect(p, (0, 0), (in_w, p.h), IN_COLOR, r, corners=(True, False, False, True))


def _io_bar_cell(in_v, out_v) -> Frame:
    """固定宽高的比例条容器，供 Grid 的 flex 布局测量。"""
    box = Frame().set_size((_IOBAR_W, 14))
    with box:
        inner = _IORatioBar(in_v, out_v, w=_IOBAR_W, h=12)
        inner.set_content_align("c")
    return box


# --------------------- 折线图 --------------------- #


def _dashed_line(d: ImageDraw.ImageDraw, pts: list, color, width: int, dash: int = 14, gap: int = 8):
    """在折线路径上画虚线：按累计弧长切段，避免两条线重合时互相遮挡。"""
    if len(pts) < 2:
        return
    period = dash + gap
    pos = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len <= 0:
            continue
        t = 0.0
        while t < seg_len:
            # 当前点在 period 内的相位，决定这一段该不该落墨
            phase = (pos + t) % period if period > 0 else 0.0
            if phase < dash:
                run = min(dash - phase, seg_len - t)
                sx = x1 + (x2 - x1) * t / seg_len
                sy = y1 + (y2 - y1) * t / seg_len
                ex = x1 + (x2 - x1) * (t + run) / seg_len
                ey = y1 + (y2 - y1) * (t + run) / seg_len
                d.line((sx, sy, ex, ey), fill=color, width=width)
                t += run
            else:
                t += min(period - phase, seg_len - t)
        pos += seg_len


def _render_line_chart(series: dict, w: int, h: int) -> Image.Image:
    """左侧轴 = Token 消耗，右侧轴 = 调用次数，两条折线各自归一化。"""
    aa = 2
    W, H = max(1, int(w * aa)), max(1, int(h * aa))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    labels = list(series.get("labels") or [])
    buckets = list(series.get("buckets") or [])
    n = len(labels)
    if n == 0 or not buckets:
        return img

    f_axis = get_font(_FONT, int(12 * aa))
    pad_l, pad_r, pad_t, pad_b = int(46 * aa), int(40 * aa), int(26 * aa), int(24 * aa)
    plot_w = max(1, W - pad_l - pad_r)
    plot_h = max(1, H - pad_t - pad_b)

    tok_top = _nice_ceil(max(b["tokens"] for b in buckets))
    call_top = _nice_ceil(max(b["calls"] for b in buckets))
    lw = max(2, int(1.7 * aa))

    def px(i: int) -> int:
        return pad_l + (plot_w // 2 if n == 1 else plot_w * i // (n - 1))

    def py(v: float, top: float) -> int:
        if top <= 0:
            return pad_t + plot_h
        return pad_t + plot_h - int(plot_h * min(1.0, max(0.0, float(v) / top)))

    tok_pts = [(px(i), py(b["tokens"], tok_top)) for i, b in enumerate(buckets)]
    call_pts = [(px(i), py(b["calls"], call_top)) for i, b in enumerate(buckets)]

    # 面积填充：在折线下方铺一层自上而下的渐隐色，让曲线有分量
    def area_layer(pts, color) -> Image.Image:
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        if n > 1:
            poly = pts + [(pts[-1][0], pad_t + plot_h), (pts[0][0], pad_t + plot_h)]
        else:
            poly = [(pts[0][0] - plot_w // 2, pad_t + plot_h), (pts[0][0] + plot_w // 2, pad_t + plot_h), pts[0]]
        ld.polygon(poly, fill=(*color[:3], 255))
        # 垂直渐变淡化
        grad = Image.new("L", (1, H))
        gp = grad.load()
        for y in range(H):
            if y < pad_t:
                gp[0, y] = 0
            else:
                t = (y - pad_t) / max(1, plot_h)
                gp[0, y] = int(70 * (1 - t))
        grad = grad.resize((W, H))
        r, g, b, a = layer.split()
        return Image.merge("RGBA", (r, g, b, ImageChops.multiply(a, grad)))

    img.alpha_composite(area_layer(tok_pts, IN_COLOR))
    img.alpha_composite(area_layer(call_pts, OUT_COLOR))

    d = ImageDraw.Draw(img)
    # 网格与坐标
    steps = 4
    for i in range(steps + 1):
        y = pad_t + plot_h * i // steps
        d.line((pad_l, y, pad_l + plot_w, y), fill=_CHART_GRID, width=max(1, aa))
        s = _fmt_token(tok_top * (steps - i) / steps)
        tw, th = get_text_size(f_axis, s)
        d.text((pad_l - int(6 * aa) - tw, y - th // 2), s, font=f_axis, fill=_CHART_TEXT)
        s = f"{call_top * (steps - i) / steps:.0f}"
        tw, th = get_text_size(f_axis, s)
        d.text((pad_l + plot_w + int(6 * aa), y - th // 2), s, font=f_axis, fill=_CHART_TEXT)

    if n > 1:
        d.line(tok_pts, fill=IN_COLOR, width=lw, joint="curve")
        _dashed_line(d, call_pts, OUT_COLOR, lw, dash=int(7 * aa), gap=int(4 * aa))
    dot = 2 * aa
    for x, y in tok_pts:
        d.ellipse((x - dot, y - dot, x + dot, y + dot), fill=IN_COLOR)
    for x, y in call_pts:
        d.ellipse((x - dot, y - dot, x + dot, y + dot), fill=OUT_COLOR)

    # 峰值标注：Token 消耗最高点画一个气泡（外圈光晕 + 内点 + 数值）
    peak_i = max(range(len(buckets)), key=lambda i: buckets[i]["tokens"])
    if buckets[peak_i]["tokens"] > 0:
        pxx, pyy = tok_pts[peak_i]
        halo = 5 * aa
        d.ellipse((pxx - halo, pyy - halo, pxx + halo, pyy + halo), fill=(*IN_COLOR[:3], 70))
        d.ellipse((pxx - 3 * aa, pyy - 3 * aa, pxx + 3 * aa, pyy + 3 * aa),
                  fill=(255, 255, 255, 235), outline=IN_COLOR, width=max(1, int(1.2 * aa)))
        peak_text = _fmt_token(buckets[peak_i]["tokens"])
        ptw, pth = get_text_size(f_axis, peak_text)
        bw, bh = ptw + int(10 * aa), pth + int(5 * aa)
        bx = min(max(pad_l, pxx - ptw // 2 - int(5 * aa)), W - pad_r - bw)
        by = pyy - bh - int(11 * aa)
        above = by >= pad_t  # 顶部放不下就翻到点的下方
        if not above:
            by = pyy + int(9 * aa)
        by = max(0, min(by, H - bh - int(2 * aa)))
        d.rounded_rectangle(
            (bx, by, bx + bw, by + bh),
            radius=int(4 * aa), fill=(*IN_COLOR[:3], 235),
        )
        d.text((bx + int(5 * aa), by + int(2.5 * aa)), peak_text, font=f_axis, fill=(255, 255, 255, 255))
        # 气泡指向峰值点的细线
        tip_x = min(max(bx + int(5 * aa), pxx), bx + bw - int(5 * aa))
        tip_y = by + bh if above else by
        d.line((tip_x, tip_y, pxx, pyy + (-halo if above else halo)), fill=(*IN_COLOR[:3], 160), width=max(1, aa))

    # X 轴标签：自适应抽稀，最多 8 个
    step_lbl = max(1, math.ceil(n / 8))
    for i in range(0, n, step_lbl):
        s = labels[i]
        tw, th = get_text_size(f_axis, s)
        x = min(max(0, px(i) - tw // 2), W - tw)
        d.text((x, H - pad_b + int(5 * aa)), s, font=f_axis, fill=_CHART_TEXT)

    # 图例：Token 消耗贴左（实线），调用次数贴右（虚线），各自对齐绘图区边缘
    ly = int(8 * aa)
    seg, gap = int(16 * aa), int(6 * aa)
    for text, color, to_right, dashed in (
        ("Token 消耗", IN_COLOR, False, False),
        ("调用次数", OUT_COLOR, True, True),
    ):
        tw, th = get_text_size(f_axis, text)
        x = (W - pad_r - seg - gap - tw) if to_right else pad_l
        if dashed:
            _dashed_line(d, [(x, ly), (x + seg, ly)], color, lw, dash=int(5 * aa), gap=int(3 * aa))
        else:
            d.line((x, ly, x + seg, ly), fill=color, width=lw)
        # 端点圆点，和图例线呼应
        d.ellipse((x - 2 * aa, ly - 2 * aa, x + 2 * aa, ly + 2 * aa), fill=color)
        d.ellipse((x + seg - 2 * aa, ly - 2 * aa, x + seg + 2 * aa, ly + 2 * aa), fill=color)
        d.text((x + seg + gap, ly - th // 2), text, font=f_axis, fill=_CHART_TEXT)

    return img.resize((int(w), int(h)), Image.Resampling.LANCZOS)


class _LineChart(Frame):
    def __init__(self, series: dict, w: int, h: int = _CHART_H):
        super().__init__()
        self.series = series
        self.set_size((w, h))
        self.add_draw_func(self._draw)

    def _draw(self, _widget, p: Painter):
        p.paste(_render_line_chart(self.series, p.w, p.h), (0, 0))


# --------------------- 双圆环 --------------------- #


def _render_donut(values: list[float], colors: list, w: int, h: int, hole: float = 0.62,
                  center_text: str = "") -> Image.Image:
    """单个圆环：values 决定扇区角度，hole 是内孔占比（0~1）。

    带柔和阴影 + 中心数值，质感和 status 的 Donut 对齐。
    """
    aa = 3
    side = max(8, int(min(w, h))) * aa
    img = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    cx = cy = side / 2.0
    radius = side * 0.5 - aa
    hole_r = radius * max(0.0, min(0.95, hole))
    dl = ImageDraw.Draw(img)
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    total = sum(v for v in values if v > 0)
    if total <= 0:
        dl.pieslice(box, -90, 270, fill=(200, 200, 200, 90))
    else:
        start = -90.0
        for v, c in zip(values, colors):
            if v <= 0:
                continue
            sweep = 360.0 * float(v) / total
            dl.pieslice(box, start, start + sweep, fill=c)
            # 相邻扇区之间留一道细缝，让分块边缘更清晰
            if sweep < 359.0:
                r0 = math.radians(start)
                dl.line(
                    [
                        (cx + hole_r * math.cos(r0), cy + hole_r * math.sin(r0)),
                        (cx + radius * math.cos(r0), cy + radius * math.sin(r0)),
                    ],
                    fill=(252, 252, 252, 190), width=max(1, int(aa * 0.9)),
                )
            start += sweep
    mask = Image.new("L", (side, side), 255)
    ImageDraw.Draw(mask).ellipse((cx - hole_r, cy - hole_r, cx + hole_r, cy + hole_r), fill=0)
    r, g, b, a = img.split()
    ring = Image.merge("RGBA", (r, g, b, ImageChops.multiply(a, mask)))

    # 阴影 + 外发光：环形 alpha 各自模糊后，先垫阴影再叠一层同色光晕
    pad = int(6 * aa)
    canvas = Image.new("RGBA", (side + pad * 2, side + pad * 2), (0, 0, 0, 0))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow.paste((20, 20, 30, 70), (pad, pad), ring.split()[3])
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=pad * 0.6))
    canvas.alpha_composite(shadow)

    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    # 用扇区颜色的平均色调做光晕，避免只反映某一个模型
    used = [c for v, c in zip(values, colors) if v > 0] or list(colors)
    if used:
        avg = tuple(sum(c[i] for c in used) // len(used) for i in range(3))
        glow.paste((*avg, 80), (pad, pad), ring.split()[3])
        glow = glow.filter(ImageFilter.GaussianBlur(radius=pad * 0.9))
        # 只保留环外扩散的部分，避免糊掉环本身的色块分界
        inner_mask = Image.new("L", canvas.size, 255)
        ImageDraw.Draw(inner_mask).ellipse(
            ((pad + cx) - radius, (pad + cy) - radius, (pad + cx) + radius, (pad + cy) + radius), fill=0,
        )
        gr, gg, gb, ga = glow.split()
        glow = Image.merge("RGBA", (gr, gg, gb, ImageChops.multiply(ga, inner_mask)))
        canvas.alpha_composite(glow)
    canvas.alpha_composite(ring, (pad, pad))

    if center_text:
        font = get_font(_BOLD, int(15 * aa))
        tw, th = get_text_size(font, center_text)
        ImageDraw.Draw(canvas).text(
            (canvas.width / 2 - tw / 2, canvas.height / 2 - th / 2),
            center_text, font=font, fill=TEXT,
        )
    out_w = max(1, int(w))
    out_h = max(1, int(h))
    canvas = canvas.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return canvas


class _Donut(Frame):
    """单个圆环 + 中心数值 + 下方标题说明。"""

    def __init__(self, title: str, values: list[float], colors: list, center_text: str = "",
                 size: int = _DONUT_SIZE):
        super().__init__()
        self.title = title
        self.values = values
        self.colors = colors
        self.center_text = center_text
        self.set_allow_draw_outside(True)
        with self:
            with VSplit().set_sep(4).set_item_align("c"):
                Frame().set_size((size, size)).set_allow_draw_outside(True).add_draw_func(self._draw_ring)
                TextBox(title, _small).set_content_align("c")

    def _draw_ring(self, _w, p: Painter):
        p.paste(_render_donut(self.values, self.colors, p.w, p.h, center_text=self.center_text), (0, 0),
                use_shadow=False)


# --------------------- 卡片 --------------------- #


def _section_head(title: str, tag_text: str = None, tag_color=GRAY_BG):
    """段标题 + 可选右侧胶囊。先构造再按实测宽度撑开中间空隙（Luna 无 FlexSpacer）。"""
    inner_w = CARD_W - 32
    with HSplit().set_w(inner_w).set_sep(8).set_item_align("c") as row:
        title_box = TextBox(title, _section)
        if tag_text:
            gap = Spacer(1, 1)
            tag = _label_box(tag_text, tag_color)
            gap.set_w(max(1, inner_w - title_box._get_self_size()[0] - tag._get_self_size()[0] - 16))
    return row


def _summary_card(start: datetime, end: datetime, total: dict):
    """头卡：左边标题 + 时间范围，右边三个汇总指标。"""
    inner_w = CARD_W - 32
    cell_w = 104
    left_w = inner_w - cell_w * 3 - _ROW_SEP * 3
    cells = (
        ("总输入", _fmt_token(total["prompt"]), IN_COLOR),
        ("总输出", _fmt_token(total["completion"]), OUT_COLOR),
        ("调用次数", _fmt_count(total["calls"]), PURPLE),
    )
    with HSplit().set_w(CARD_W).set_padding(16).set_sep(_ROW_SEP).set_item_align("c").set_bg(_card_bg()) as card:
        with VSplit().set_w(left_w).set_sep(8).set_item_align("l"):
            TextBox("Token 用量统计", _title)
            with Flow(hsep=6, vsep=6).set_w(left_w):
                range_text = _fmt_range(start, end)
                # 直接量文本宽度定宽（widget 的 _get_self_size 有缓存，先测后 set_w 会失效）
                text_w = get_text_size(get_font(_FONT, 15), range_text)[0] + 14
                _label_box(range_text, GRAY_BG).set_w(min(left_w, text_w)).set_allow_draw_outside(True)
        for label, value, color in cells:
            with VSplit().set_w(cell_w).set_sep(8).set_item_align("c"):
                _label_box(label, color).set_allow_draw_outside(True)
                TextBox(value, _big).set_content_align("c")
    return card


def _chart_card(series: dict):
    inner_w = CARD_W - 32
    with VSplit().set_w(CARD_W).set_padding(16).set_sep(10).set_item_align("l").set_bg(_card_bg()) as card:
        _LineChart(series, inner_w, _CHART_H)
    return card


def _split_top_models(models: list[dict]) -> list[dict]:
    """按总消耗排名，前 N 个单独上色，其余合并成「其它模型」（灰色）。

    每个模型的颜色从调色板里随机抽且不重复；模型数超出调色板时才允许重复取色。
    「其它模型」固定用灰色，不参与抽取。
    """
    try:
        top_n = int(stat_config.get("chat_top_n", 3) or 3)
    except (TypeError, ValueError):
        top_n = 3
    other_label = str(stat_config.get("other_label", "") or "其它模型")
    top_n = max(1, top_n)

    limit = min(len(models), top_n)
    # 随机打乱调色板后按序取用；不够则重新打乱续接，保证同色尽量不相邻
    palette = list(_MODEL_COLORS)
    random.shuffle(palette)
    colors = palette * ((limit + len(palette) - 1) // len(palette))
    colors = colors[:limit]

    entries = [{**row, "color": colors[i]} for i, row in enumerate(models[:limit])]
    rest = models[top_n:]
    if rest:
        entries.append({
            "label": other_label,
            "color": OTHER_COLOR,
            "prompt": sum(r["prompt"] for r in rest),
            "completion": sum(r["completion"] for r in rest),
            "calls": sum(r["calls"] for r in rest),
            "merged": True,
        })
    return entries


def _model_detail_table(entries: list[dict], inner_w: int):
    """参照 status._metric_table 的对齐手法：标签与数值各自成列，
    列宽取该列最宽格子、数值右对齐，左侧名字列吃掉剩余宽度。"""
    vsep = 12
    with HSplit().set_w(inner_w).set_sep(8).set_item_align("t").set_padding(0) as block:
        with VSplit().set_sep(vsep).set_item_align("l").set_padding(0) as names_col:
            name_boxes = [
                TextBox(
                    e["label"], TextStyle(font=_FONT, size=16, color=e["color"]),
                    wrap=False, overflow="shrink",
                ).set_padding(0)
                for e in entries
            ]
        with (
            Grid(col_count=7)
            .set_item_size_mode("flex")
            .set_item_align("r")
            .set_sep(hsep=8, vsep=vsep)
            .set_padding(0) as grid
        ):
            for e in entries:
                TextBox("输入", _dim, wrap=False).set_padding(0)
                TextBox(_fmt_token(e["prompt"]), _small, wrap=False).set_padding(0)
                _io_bar_cell(e["prompt"], e["completion"])
                TextBox("输出", _dim, wrap=False).set_padding(0)
                TextBox(_fmt_token(e["completion"]), _small, wrap=False).set_padding(0)
                TextBox("·", _dim, wrap=False).set_padding(0)
                TextBox(f"{_fmt_count(e['calls'])}次", _small, wrap=False).set_padding(0)
        cap = max(1, inner_w - grid._get_self_size()[0] - 8)
        names_col.set_w(cap)
        for box in name_boxes:
            box.set_w(cap)
    return block


def _models_card(models: list[dict]):
    inner_w = CARD_W - 32
    entries = _split_top_models(models)
    with VSplit().set_w(CARD_W).set_padding(16).set_sep(10).set_item_align("l").set_bg(_card_bg()) as card:
        if not entries:
            TextBox("(无记录)", _small)
            return card
        # 两个圆环并列：左=Token 消耗占比，右=调用次数占比
        tk_total = sum(e["prompt"] + e["completion"] for e in entries)
        call_total = sum(e["calls"] for e in entries)
        with HSplit().set_w(inner_w).set_sep(16).set_item_align("t").set_item_size_mode("expand") as charts:
            _Donut(
                "Token 消耗", [e["prompt"] + e["completion"] for e in entries],
                [e["color"] for e in entries], center_text=_fmt_token(tk_total),
            )
            _Donut(
                "调用次数", [e["calls"] for e in entries],
                [e["color"] for e in entries], center_text=_fmt_count(call_total),
            )
        _divider(inner_w)
        _model_detail_table(entries, inner_w)
    return card


def _extra_value_text(e: dict) -> str:
    if e["kind"] == "tts":
        return f"{_fmt_count(e['prompt'])} 字 · {_fmt_count(e['calls'])} 次"
    return f"输入 {_fmt_token(e['prompt'])} · {_fmt_count(e['calls'])} 次"


def _extra_kind_tag(e: dict) -> TextBox:
    """emb / tts 的类型胶囊。"""
    if e["kind"] == "tts":
        return _label_box("TTS", ORANGE, w=54)
    return _label_box("EMB", BLUE, w=54)


def _extras_card(extras: list[dict]):
    inner_w = CARD_W - 32
    value_w = 156
    tag_w = 54
    with VSplit().set_w(CARD_W).set_padding(16).set_sep(10).set_item_align("l").set_bg(_card_bg()) as card:
        _section_head("其它类型模型", None, ORANGE)
        _divider(inner_w)
        if not extras:
            TextBox("(无记录)", _small)
        for e in extras:
            with HSplit().set_w(inner_w).set_sep(_ROW_SEP).set_item_align("c"):
                _extra_kind_tag(e).set_allow_draw_outside(True)
                TextBox(e["label"], _body, wrap=False, overflow="shrink").set_w(inner_w - value_w - tag_w - _ROW_SEP * 2)
                TextBox(_extra_value_text(e), _small, wrap=False).set_w(value_w).set_content_align("r")
    return card


async def draw_token_report(start: datetime, end: datetime, data: dict):
    """画一张统计卡片，返回 PIL Image。"""
    chat = data["chat"]
    with Canvas(w=CANVAS_W, bg=_canvas_bg()).set_padding(0) as canvas:
        with VSplit().set_w(CANVAS_W).set_padding(16).set_sep(14).set_item_align("c").set_bg(FillBg(MASK)):
            _summary_card(start, end, chat["total"])
            _chart_card(chat["series"])
            _models_card(chat["by_model"])
            _extras_card(data["extras"])

    img = await canvas.get_img(scale=2)
    return _stamp_watermark(img, _WATERMARK)


def _stamp_watermark(img: Image.Image, text: str) -> Image.Image:
    """右下角水印：浅色小字 + 轻微阴影。用字体 bbox 精确定位，落在卡片下沿与图底之间。"""
    if not text:
        return img
    img = img.convert("RGBA")
    w, h = img.size
    # 字号随图宽缩放，但保证不小于 _WATERMARK_SIZE
    font = get_font(_FONT, max(_WATERMARK_SIZE, w // 62))
    # 用 bbox 而不是 get_text_size：后者含行距，会导致定位偏上
    x0, y0, x1, y1 = font.getbbox(text)
    tw, th = x1 - x0, y1 - y0
    pad_x, pad_bottom = max(12, w // 80), 4
    # 反推 draw.text 的锚点：墨迹顶部 = y + y0，墨迹底边 = y + y1
    x = w - pad_x - tw - x0
    y = h - pad_bottom - th - y0

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    # 阴影垫一层，浅色背景上也能看清
    ld.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 80))
    ld.text((x, y), text, font=font, fill=(255, 255, 255, 170))
    return Image.alpha_composite(img, layer).convert("RGB")


# ======================= 指令处理 ======================= #

token_stat = CmdHandler(["/token stat", "/tokens", "/ts"], logger)
token_stat.check_cdrate(cd)


@token_stat.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()
    try:
        start, end = parse_token_range(args)
    except ValueError as e:
        return await ctx.asend_reply_msg(str(e))
    data = await query_token_usage(start, end)
    if data["total"]["calls"] == 0:
        return await ctx.asend_reply_msg(f"{_fmt_range(start, end)} 没有 token 用量记录")
    img = await draw_token_report(start, end, data)
    return await ctx.asend_reply_msg(await get_image_cq(img, low_quality=True))
