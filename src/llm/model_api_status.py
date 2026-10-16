# Model API 状态监控：统计各模型 API 调用成功率与响应时间，画时间线卡片
import asyncio
import math
from datetime import datetime, timedelta

from src.utils import *

from .sql import _get_conn, register_on_record, get_recent_calls, set_cache_max, flush_pending
from .draw import (
    CARD_W, CANVAS_W, MASK,
    _FONT, _small, _dim, _model_name,
    _canvas_bg, _card_bg, _label_box, _round_rect,
    _fmt_count, _fmt_range, _make_time_buckets, parse_token_range,
    _stamp_watermark, _WATERMARK, cfg_get,
)

logger = get_logger("llm")
file_db = get_file_db("data/llm/db.json")
cd = ColdDown(file_db, logger, cold_down_name="model_api_status")
config = Config("llm.llm")

# 时间线四态：绿=成功且不慢、黄=成功但慢、红=失败、灰=无请求
_ST_NONE, _ST_OK, _ST_SLOW, _ST_FAIL = "none", "ok", "slow", "fail"
_TL_COLORS = {
    _ST_OK: (46, 178, 90, 255),
    _ST_SLOW: (240, 173, 60, 255),
    _ST_FAIL: (224, 86, 97, 255),
    _ST_NONE: (214, 217, 222, 255),
}

# 成功率 → 状态标签（✓ / ⚠ 需字体支持，思源黑体实测可用）
_STATUS_OK, _STATUS_WARN, _STATUS_BAD = "ok", "warn", "bad"
_STATUS_STYLE = {
    _STATUS_OK: ("✓ 正常", (29, 169, 18, 210)),
    _STATUS_WARN: ("⚠ 警告", (238, 144, 37, 210)),
    _STATUS_BAD: ("⚠ 异常", (224, 86, 97, 210)),
}
_STATUS_ORDER = {_STATUS_BAD: 0, _STATUS_WARN: 1, _STATUS_OK: 2}

# 推送状态（存 db，跨重启保留）
_NOTIFY_BAD, _NOTIFY_OK = "bad", "ok"

_AXIS_TEXT = (108, 114, 126, 255)
_KIND_STYLE = {
    "chat": ("CHAT", (29, 169, 18, 170)),
    "embedding": ("EMB", (17, 141, 195, 170)),
    "tts": ("TTS", (238, 144, 37, 170)),
}


# ======================= 查询 ======================= #


def _build_timeline(calls: list, start, end, slow_ms: int):
    """把逐次调用铺到时间桶上，每桶取该段内最差状态。

    calls: [(ts, ok, latency_ms), ...]
    """
    base = _make_time_buckets(start, end)
    labels, step = base["labels"], base["step"]
    states = [_ST_NONE] * len(labels)
    for ts, ok, lat in calls:
        idx = int((float(ts) - start.timestamp()) // step)
        if not (0 <= idx < len(states)):
            continue
        cur = states[idx]
        if not ok:
            states[idx] = _ST_FAIL
        elif int(lat or 0) > slow_ms:
            if cur != _ST_FAIL:
                states[idx] = _ST_SLOW
        elif cur == _ST_NONE:
            states[idx] = _ST_OK
    return states, labels, base.get("end_label")


async def query_api_status(start, end) -> dict:
    """聚合 [start, end) 的调用记录，按 provider:model:kind 分组。"""
    conn = await _get_conn()
    cursor = await conn.execute(
        "SELECT provider, model, kind, time, ok, latency_ms, error"
        " FROM llm_calls WHERE time >= ? AND time < ? ORDER BY time",
        (start.timestamp(), end.timestamp()),
    )
    rows = await cursor.fetchall()
    await cursor.close()

    slow_ms = cfg_get("model_api_status.slow_ms", int)
    ok_rate = cfg_get("model_api_status.ok_rate", float)
    warn_rate = cfg_get("model_api_status.warn_rate", float)

    groups: dict[tuple, dict] = {}
    for provider, model, kind, ts, ok, lat, err in rows:
        key = (str(provider), str(model), str(kind))
        g = groups.setdefault(key, {
            "provider": str(provider), "model": str(model), "kind": str(kind),
            "total": 0, "ok": 0, "fail": 0, "lat_sum": 0, "lat_n": 0,
            "calls": [], "last_error": None,
        })
        ok, lat = int(ok), int(lat or 0)
        g["total"] += 1
        if ok:
            g["ok"] += 1
            g["lat_sum"] += lat
            g["lat_n"] += 1
        else:
            g["fail"] += 1
            if err:
                g["last_error"] = str(err)
        g["calls"].append((float(ts), ok, lat))

    items = []
    for g in groups.values():
        rate = g["ok"] / g["total"] if g["total"] else 0.0
        if rate >= ok_rate:
            status = _STATUS_OK
        elif rate >= warn_rate:
            status = _STATUS_WARN
        else:
            status = _STATUS_BAD
        states, labels, end_label = _build_timeline(g["calls"], start, end, slow_ms)
        items.append({
            "provider": g["provider"], "model": g["model"], "kind": g["kind"],
            "total": g["total"], "ok": g["ok"], "fail": g["fail"],
            "rate": rate, "status": status,
            "avg_ms": (g["lat_sum"] / g["lat_n"]) if g["lat_n"] else 0,
            "states": states, "labels": labels, "end_label": end_label,
            "last_error": g["last_error"],
        })

    # 异常优先，其次按请求量降序
    items.sort(key=lambda it: (_STATUS_ORDER.get(it["status"], 3), -it["total"]))
    return {"items": items, "slow_ms": slow_ms, "ok_rate": ok_rate, "warn_rate": warn_rate}


# ======================= 异常推送 ======================= #


async def _fetch_recent_calls(provider: str, model: str, kind: str, limit: int) -> list[dict]:
    """取某模型最近 limit 次调用，按时间正序返回（最早在前）。读内存缓存，不回查磁盘。"""
    return await get_recent_calls(provider, model, kind, limit)


def _judge_fail(calls: list[dict], rule: dict) -> str | None:
    """判定是否异常，命中则返回原因描述。calls 按时间正序。

    规则：连续 N 次失败，或最近 M 次里失败 ≥K 次。
    """
    streak = int(rule["fail_streak"])
    window = int(rule["fail_window"])
    threshold = int(rule["fail_count"])

    # 连续失败
    if streak > 0 and len(calls) >= streak:
        tail = calls[-streak:]
        if all(not c["ok"] for c in tail):
            return f"连续 {streak} 次失败"

    # 滑窗失败数
    if window > 0 and len(calls) >= window:
        tail = calls[-window:]
        fails = sum(1 for c in tail if not c["ok"])
        if fails >= threshold:
            return f"最近 {window} 次失败 {fails} 次"
    return None


def _judge_recover(calls: list[dict], rule: dict) -> str | None:
    """判定是否恢复，命中则返回原因描述。"""
    streak = int(rule["recover_streak"])
    window = int(rule["recover_window"])
    threshold = int(rule["recover_count"])

    if streak > 0 and len(calls) >= streak:
        tail = calls[-streak:]
        if all(c["ok"] for c in tail):
            return f"连续 {streak} 次成功"

    if window > 0 and len(calls) >= window:
        tail = calls[-window:]
        oks = sum(1 for c in tail if c["ok"])
        if oks >= threshold:
            return f"最近 {window} 次成功 {oks} 次"
    return None


def _notify_cfg() -> dict:
    return {
        "fail_streak": cfg_get("model_api_status.notify.fail_streak", int),
        "fail_window": cfg_get("model_api_status.notify.fail_window", int),
        "fail_count": cfg_get("model_api_status.notify.fail_count", int),
        "recover_streak": cfg_get("model_api_status.notify.recover_streak", int),
        "recover_window": cfg_get("model_api_status.notify.recover_window", int),
        "recover_count": cfg_get("model_api_status.notify.recover_count", int),
        "eval_limit": cfg_get("model_api_status.notify.eval_limit", int),
        "chart_max_span": cfg_get("model_api_status.notify.chart_max_span", int),
    }


def _state_key(provider: str, model: str, kind: str) -> str:
    # FileDB 按 "." 拆层级，模型名里常带 "."（如 glm-4.6v-flash）；
    # 用 \. 转义让点号保持字面量，避免被误拆成嵌套结构
    def esc(s) -> str:
        return str(s).replace(".", r"\.")
    return f"notify_states.{esc(provider)}.{esc(model)}.{esc(kind)}"


# 按模型粒度的评估锁：防止并发调用触发重复推送
_eval_locks: dict[str, asyncio.Lock] = {}


def _eval_lock(key: str) -> asyncio.Lock:
    lock = _eval_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _eval_locks[key] = lock
    return lock


async def _draw_single_model(provider: str, model: str, kind: str, start, end) -> bytes:
    """按指定时间范围画单个模型的卡片，返回图片。"""
    data = await query_api_status(start, end)
    item = next(
        (it for it in data["items"]
         if it["provider"] == provider and it["model"] == model and it["kind"] == kind),
        None,
    )
    if item is None:
        return None
    # 复用汇总卡绘制流程，但只画这一个模型
    with Canvas(w=CANVAS_W, bg=_canvas_bg()).set_padding(0) as canvas:
        with VSplit().set_w(CANVAS_W).set_padding(16).set_sep(14).set_item_align("c").set_bg(FillBg(MASK)):
            _model_card(item, start, end)
    img = await canvas.get_img(scale=2)
    return _stamp_watermark(img, _WATERMARK)


async def _evaluate_and_notify(provider: str, model: str, kind: str) -> None:
    """事件驱动：一次调用落库后评估该模型，必要时推送异常/恢复。"""
    groups = notify_gwl.get()
    if not groups:
        return

    rule = _notify_cfg()
    limit = max(
        rule["eval_limit"],
        rule["fail_streak"], rule["fail_window"],
        rule["recover_streak"], rule["recover_window"],
    )
    key = _state_key(provider, model, kind)
    label = f"{provider}:{model} [{kind}]"

    async with _eval_lock(key):
        calls = await _fetch_recent_calls(provider, model, kind, limit)
        if not calls:
            return

        prev = str(file_db.get(key, "") or "")
        if prev != _NOTIFY_BAD:
            reason = _judge_fail(calls, rule)
            if reason is None:
                return
            state, title = _NOTIFY_BAD, "⚠模型API异常通知"
        else:
            reason = _judge_recover(calls, rule)
            if reason is None:
                return
            state, title = _NOTIFY_OK, "✓模型API恢复通知"
        # 先落状态再推送：并发评估时后到的任务会直接跳过，避免重复刷屏
        file_db.set(key, state)

    # 慢操作放在锁外：画图 + 推送
    # 画图从 sqlite 取数，触发评估的调用可能还压在待落盘队列里，先强制落一次
    try:
        await flush_pending()
    except Exception as e:
        logger.warning("推送前落盘失败: %s", get_exc_desc(e))

    end = datetime.now()
    first_ts = min(c["time"] for c in calls)
    span_limit = end - timedelta(seconds=int(rule["chart_max_span"]))
    start = max(datetime.fromtimestamp(first_ts), span_limit)

    try:
        img = await _draw_single_model(provider, model, kind, start, end)
    except Exception as e:
        logger.print_exc(f"生成 {label} 状态图失败: {get_exc_desc(e)}")
        img = None

    cq = await get_image_cq(img, low_quality=True) if img is not None else ""
    headline = title
    state_name = "异常" if state == _NOTIFY_BAD else "恢复"
    logger.info("API状态推送 %s: %s (%s)", state_name, label, reason)

    for group_id in groups:
        try:
            # get_image_cq 返回的是完整 CQ 码，直接与文本拼接即可
            msg = f"{headline}\n{cq}" if cq else headline
            await send_group_msg_by_bot(group_id, msg)
        except Exception as e:
            logger.print_exc(f"向群 {group_id} 推送 {label} 状态失败: {get_exc_desc(e)}")


# ======================= 绘图 ======================= #


def _fmt_ms(ms) -> str:
    v = float(ms or 0)
    if v < 1000:
        return f"{v:.0f}ms"
    if v < 60000:
        return f"{v / 1000:.1f}s"
    return f"{v / 60000:.1f}min"


class _Timeline(Frame):
    """状态时间线条：每格代表一个时间段，格间留缝。"""

    BAR_H = 26

    def __init__(self, states: list, labels: list, w: int, end_label: str = None):
        super().__init__()
        self.states = states
        self.labels = labels
        self.end_label = end_label
        self.set_size((w, self.BAR_H + 20))
        self.add_draw_func(self._draw)

    def _draw(self, _widget, p: Painter):
        n = len(self.states)
        if n == 0:
            return
        gap = 1 if n <= 60 else 0
        total_gap = gap * (n - 1)
        cell_w = max(1, (p.w - total_gap) // n)
        radius = max(1, min(3, cell_w // 3))
        for i, st in enumerate(self.states):
            x = i * (cell_w + gap)
            _round_rect(p, (x, 0), (cell_w, self.BAR_H), _TL_COLORS.get(st, _TL_COLORS[_ST_NONE]), radius)

        # 时间刻度：抽稀到最多 8 个；右端固定标出"当前时间"
        font = get_font(_FONT, 12)
        end_x = None
        if self.end_label:
            tw, _th = get_text_size(font, self.end_label)
            end_x = max(0, p.w - tw)
        step_lbl = max(1, math.ceil(n / 8))
        for i in range(0, n, step_lbl):
            s = self.labels[i]
            tw, _th = get_text_size(font, s)
            x = min(max(0, i * (cell_w + gap) + cell_w // 2 - tw // 2), max(0, p.w - tw))
            if end_x is not None and x + tw > end_x - 6:
                continue
            p.text(s, (x, self.BAR_H + 4), font, fill=_AXIS_TEXT)
        if end_x is not None:
            p.text(self.end_label, (end_x, self.BAR_H + 4), font, fill=_AXIS_TEXT)


def _title_row(item: dict, inner_w: int):
    """左侧类型胶囊 + 模型名（大字），右侧状态胶囊贴卡片右缘。"""
    with HSplit().set_w(inner_w).set_sep(8).set_item_align("c") as row:
        kind_text, kind_color = _KIND_STYLE.get(item["kind"], ("CHAT", (17, 141, 195, 170)))
        kind_box = _label_box(kind_text, kind_color)
        name_box = TextBox(f"{item['provider']}:{item['model']}", _model_name, overflow="shrink")
        # Spacer 必须放在状态胶囊之前：中间吃满剩余宽度，状态胶囊才能贴到右缘
        gap = Spacer(1, 1)
        st_text, st_color = _STATUS_STYLE[item["status"]]
        st_box = _label_box(st_text, st_color)
        fixed = (kind_box._get_self_size()[0] + name_box._get_self_size()[0]
                 + st_box._get_self_size()[0] + 24)
        gap.set_w(max(1, inner_w - fixed))
    return row


def _model_card(item: dict, start, end) -> Frame:
    inner_w = CARD_W - 32
    with VSplit().set_w(CARD_W).set_padding(16).set_sep(10).set_item_align("l").set_bg(_card_bg()) as card:
        _title_row(item, inner_w)
        TextBox(f"总请求: {_fmt_count(item['total'])} 次", _small)
        TextBox(
            f"成功率: {item['rate'] * 100:.1f}%      成功: {_fmt_count(item['ok'])} 次      失败: {_fmt_count(item['fail'])} 次",
            _small,
        )
        _Timeline(item["states"], item["labels"], inner_w, item.get("end_label"))
        TextBox(f"平均请求时间: {_fmt_ms(item['avg_ms'])}", _small)
        if item["last_error"]:
            TextBox(f"最近错误: {truncate(item['last_error'], 90)}", _dim, overflow="shrink").set_w(inner_w)
    return card


async def draw_api_report(start, end, data: dict):
    """画 API 状态卡片，每模型一张。"""
    items = data["items"]
    max_show = cfg_get("model_api_status.max_show", int)
    with Canvas(w=CANVAS_W, bg=_canvas_bg()).set_padding(0) as canvas:
        with VSplit().set_w(CANVAS_W).set_padding(16).set_sep(14).set_item_align("c").set_bg(FillBg(MASK)):
            for it in items[:max_show]:
                _model_card(it, start, end)
            if len(items) > max_show:
                TextBox(f"(仅显示前 {max_show} 项，共 {len(items)} 项)", _dim).set_w(CARD_W).set_content_align("c")

    img = await canvas.get_img(scale=2)
    return _stamp_watermark(img, _WATERMARK)


# ======================= 指令处理 ======================= #

api_status = CmdHandler(["/model api status", "/mas", "/ma status"], logger)
api_status.check_cdrate(cd)


@api_status.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()
    try:
        start, end = parse_token_range(args)
    except ValueError as e:
        return await ctx.asend_reply_msg(str(e))
    data = await query_api_status(start, end)
    if not data["items"]:
        return await ctx.asend_reply_msg(f"{_fmt_range(start, end)} 没有 API 调用记录")
    img = await draw_api_report(start, end, data)
    return await ctx.asend_reply_msg(await get_image_cq(img, low_quality=True))


# ======================= 异常推送 ======================= #

# 群白名单：/model_api_status_notify on|off|status
notify_gwl = get_group_white_list(file_db, logger, "model_api_status_notify", is_service=False)

# 内存缓存条数 = 评估窗口 * 3，足够应对并发写入
set_cache_max(cfg_get("model_api_status.notify.eval_limit", int) * 3)

# 注册事件钩子：每次调用入内存后评估该模型
register_on_record(_evaluate_and_notify)