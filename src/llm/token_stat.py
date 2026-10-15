# Token 用量统计：/token stat 查询，数据来自 sql.py 的 llm_calls 表
from src.utils import *

from .sql import _get_conn
from .draw import (
    CARD_W, CANVAS_W, MASK, GRAY_BG, PURPLE, BLUE, ORANGE, OTHER_COLOR,
    IN_COLOR, OUT_COLOR, _MODEL_COLORS,
    _BOLD, _FONT, _body, _small, _dim, _label, _big, _title,
    _canvas_bg, _card_bg, _label_box, _divider, _section_head,
    _fmt_count, _fmt_token, _io_bar_cell, _LineChart, _Donut,
    _stamp_watermark, _WATERMARK, _CHART_H, _ROW_SEP, cfg_get,
    parse_token_range, _fmt_range, _make_time_buckets,
)

logger = get_logger("llm")
file_db = get_file_db("data/llm/db.json")
cd = ColdDown(file_db, logger, cold_down_name="token_stat")
config = Config("llm.llm")

_CARD_SEP = 14


# ======================= 查询 ======================= #


async def query_token_usage(start, end) -> dict:
    """聚合 [start, end) 范围内的用量（只统计成功的调用）。

    返回：
      total      —— 全部调用的合计（prompt/completion/calls）
      chat       —— 只含 chat：{total, by_model(带输入输出), series(按时序)}
      extras     —— embedding / tts 明细列表（每条含 kind）
    """
    conn = await _get_conn()
    cursor = await conn.execute(
        """
        SELECT provider, model, kind,
               SUM(prompt_tokens), SUM(completion_tokens), COUNT(*)
        FROM llm_calls
        WHERE time >= ? AND time < ? AND ok = 1
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

    # 时序（只统计 chat）
    cursor = await conn.execute(
        """
        SELECT time, prompt_tokens, completion_tokens
        FROM llm_calls
        WHERE time >= ? AND time < ? AND ok = 1 AND kind = 'chat'
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


# ======================= 绘图 ======================= #


def _summary_card(start, end, total: dict):
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
                # 限宽 + shrink：数值太长（如 10.68M）时避免溢出报错
                TextBox(value, _big, overflow="shrink").set_w(cell_w).set_content_align("c")
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
    import random
    top_n = cfg_get("token_stat.chat_top_n", int)
    other_label = cfg_get("token_stat.other_label", str)
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


async def draw_token_report(start, end, data: dict):
    """画一张统计卡片，返回 PIL Image。"""
    chat = data["chat"]
    with Canvas(w=CANVAS_W, bg=_canvas_bg()).set_padding(0) as canvas:
        with VSplit().set_w(CANVAS_W).set_padding(16).set_sep(_CARD_SEP).set_item_align("c").set_bg(FillBg(MASK)):
            _summary_card(start, end, chat["total"])
            _chart_card(chat["series"])
            _models_card(chat["by_model"])
            _extras_card(data["extras"])

    img = await canvas.get_img(scale=2)
    return _stamp_watermark(img, _WATERMARK)


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