import re
from datetime import timedelta

from src.utils import *

_BYTE_UNITS = ("B", "K", "M", "G", "T", "P")


def match_list_regexp(reg_list: list[str], txt: str) -> re.Match | None:
    """用一组正则去匹配文本，返回第一个命中。"""
    for pattern in reg_list or []:
        try:
            m = re.search(pattern, txt)
        except re.error:
            continue
        if m:
            return m
    return None


def auto_convert_byte(value: float, suffix: str = "", unit_index: int = 0, with_space: bool = False) -> str:
    """把字节数换成可读单位。"""
    v = float(value or 0)
    i = unit_index
    while abs(v) >= 1024 and i < len(_BYTE_UNITS) - 1:
        v /= 1024
        i += 1
    if abs(v) >= 100 or i == unit_index:
        num = f"{v:.0f}"
    else:
        num = f"{v:.1f}"
    space = " " if with_space else ""
    return f"{num}{space}{_BYTE_UNITS[i]}{suffix}"


def format_cpu_freq(current: float | None, maximum: float | None) -> str:
    """格式化 CPU 主频。"""
    def hz(v: float) -> str:
        return auto_convert_byte(v, suffix="Hz", unit_index=2, with_space=False)

    if not current:
        return "主频未知"
    if not maximum or maximum == current:
        return hz(current)
    return f"{hz(current)} / {hz(maximum)}"


def format_uptime(delta: timedelta) -> str:
    """格式化运行时长。"""
    if delta.total_seconds() < 0:
        delta = timedelta(0)
    return get_readable_timedelta(delta, precision="m")


def percent_color(percent: float | None) -> tuple[int, int, int, int]:
    """按占用百分比选状态图颜色。"""
    if percent is None:
        return (190, 190, 190, 170)
    if percent < 70:
        return (29, 169, 18, 170)
    if percent < 90:
        return (238, 144, 37, 170)
    return (224, 86, 97, 170)
