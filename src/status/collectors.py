import asyncio
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from src.utils import *

from .util import format_cpu_freq, format_uptime, match_list_regexp

config = Config("status")
logger = get_logger("status")

recv_num: dict[str, int] = {}
send_num: dict[str, int] = {}
bot_nick_cache: dict[str, str] = {}
bot_avatar_cache: dict[str, Any] = {}
started_at = datetime.now()

_cpu_brand: str | None = None
_system_name: str | None = None


@dataclass
class BotStatus:
    self_id: str
    adapter: str
    nick: str
    bot_connected: str
    msg_rec: str
    msg_sent: str
    avatar: Any = None


@dataclass
class CpuMem:
    cpu_percent: float
    cpu_count: int | None
    cpu_count_logical: int | None
    cpu_freq: str
    cpu_brand: str
    ram_percent: float
    ram_used: int
    ram_total: int
    swap_percent: float | None
    swap_used: int
    swap_total: int


@dataclass
class DiskUsage:
    name: str
    percent: float | None = None
    used: int = 0
    total: int = 0
    exception: str | None = None


@dataclass
class NamedRate:
    name: str
    read: float = 0.0
    write: float = 0.0
    sent: float = 0.0
    recv: float = 0.0


@dataclass
class SiteResult:
    name: str
    status: int | None = None
    reason: str = ""
    delay: float | None = None
    error: str | None = None


@dataclass
class ProcStatus:
    name: str
    cpu: float
    mem: int


@dataclass
class StatusData:
    bots: list[BotStatus] = field(default_factory=list)
    bot_run_time: str = ""
    system_run_time: str = ""
    cpu_mem: CpuMem | None = None
    disk_usage: list[DiskUsage] = field(default_factory=list)
    disk_io: list[NamedRate] = field(default_factory=list)
    network_io: list[NamedRate] = field(default_factory=list)
    network_connection: list[SiteResult] = field(default_factory=list)
    process_status: list[ProcStatus] = field(default_factory=list)
    time: str = ""
    python_version: str = ""
    system_name: str = ""


def note_recv(self_id: int | str) -> None:
    key = str(self_id)
    recv_num[key] = recv_num.get(key, 0) + 1


def note_send(self_id: int | str) -> None:
    key = str(self_id)
    send_num[key] = send_num.get(key, 0) + 1


def ensure_counters(self_id: int | str) -> None:
    key = str(self_id)
    recv_num.setdefault(key, 0)
    send_num.setdefault(key, 0)


def drop_counters(self_id: int | str) -> None:
    key = str(self_id)
    if config.get("disconnect_reset_counter", True):
        recv_num.pop(key, None)
        send_num.pop(key, None)
        bot_nick_cache.pop(key, None)
        bot_avatar_cache.pop(key, None)


def _cpu_brand_sync() -> str:
    global _cpu_brand
    if _cpu_brand is not None:
        return _cpu_brand
    try:
        from cpuinfo import get_cpu_info

        brand = str(get_cpu_info().get("brand_raw") or "").split("@", maxsplit=1)[0].strip()
        if brand.lower().endswith(("cpu", "processor")):
            brand = brand.rsplit(maxsplit=1)[0].strip()
        _cpu_brand = brand or "未知型号"
    except Exception:
        logger.print_exc("读取 CPU 型号失败")
        _cpu_brand = "未知型号"
    return _cpu_brand


def _parse_env_file(path: str) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().upper()] = v.strip().strip("\"'")
    return out


def _system_name_sync() -> str:
    global _system_name
    if _system_name is not None:
        return _system_name
    system, _, release, version, machine, _ = platform.uname()
    system, release, version = platform.system_alias(system, release, version)
    if system == "Darwin":
        _system_name = f"MacOS {platform.mac_ver()[0]} {machine}"
        return _system_name
    if system == "Windows":
        edition = ""
        try:
            edition = platform.win32_edition() or ""
        except Exception:
            pass
        _system_name = f"Windows {release} {edition} {machine}".strip()
        return _system_name
    if system == "Linux":
        if (pfx := os.getenv("PREFIX")) and "termux" in pfx:
            _system_name = f"Termux (Android) {release} {machine}"
            return _system_name
        if os.getenv("ANDROID_ROOT") == "/system":
            _system_name = f"Linux (Android) {release} {machine}"
            return _system_name
        env = _parse_env_file("/etc/os-release") or _parse_env_file("/etc/lsb-release")
        name = env.get("NAME") or env.get("DISTRIB_ID")
        ver = env.get("VERSION_ID") or env.get("DISTRIB_RELEASE")
        if name and ver:
            shown = release if ver.lower() == "rolling" else ver
            _system_name = f"{name} {shown} {machine}"
        else:
            _system_name = f"未知 Linux {release} {machine}"
        return _system_name
    _system_name = f"{system} {release}"
    return _system_name


def _disk_usage_sync() -> list[DiskUsage]:
    ignore_parts = list(config.get("ignore_parts", []) or [])
    ignore_bad = bool(config.get("ignore_bad_parts", False))
    usage: list[DiskUsage] = []
    try:
        parts = psutil.disk_partitions()
    except Exception:
        logger.debug("读取磁盘分区失败", exc_info=True)
        return usage
    for disk in parts:
        mountpoint = disk.mountpoint
        if match_list_regexp(ignore_parts, mountpoint):
            continue
        try:
            u = psutil.disk_usage(mountpoint)
        except Exception as e:
            if ignore_bad:
                continue
            usage.append(DiskUsage(name=mountpoint, exception=str(e)))
            continue
        usage.append(DiskUsage(name=mountpoint, percent=u.percent, used=u.used, total=u.total))
    if config.get("sort_parts", True):
        reverse = not bool(config.get("sort_parts_reverse", False))
        usage.sort(key=lambda x: x.percent if x.percent is not None else -1, reverse=reverse)
    return usage


def _process_status_sync(procs: list[psutil.Process]) -> list[ProcStatus]:
    limit = int(config.get("proc_len", 5) or 0)
    if limit <= 0:
        return []
    ignore = list(config.get("ignore_procs", []) or [])
    cpu_count = psutil.cpu_count() or 1
    max_100 = bool(config.get("proc_cpu_max_100p", False))
    items: list[ProcStatus] = []
    for proc in procs:
        try:
            name = proc.name()
            if match_list_regexp(ignore, name):
                continue
            # cpu_percent 不能放进 oneshot：oneshot 会缓存同一份 cpu_times，第二次采样算不出 delta
            cpu = proc.cpu_percent()
            mem = proc.memory_info().rss
            if mem <= 0:
                continue
            if max_100:
                cpu = cpu / cpu_count
            items.append(ProcStatus(name=name, cpu=cpu, mem=mem))
        except (psutil.Error, OSError):
            continue
    sort_by = str(config.get("proc_sort_by", "cpu") or "cpu").lower()
    if sort_by in ("mem", "men", "memory"):
        items.sort(key=lambda x: x.mem, reverse=True)
    else:
        items.sort(key=lambda x: x.cpu, reverse=True)
    return items[:limit]


def _calc_disk_io(past, now, dt: float) -> list[NamedRate]:
    ignore = list(config.get("ignore_disk_ios", []) or [])
    ignore_zero = bool(config.get("ignore_no_io_disk", False))
    out: list[NamedRate] = []
    for name, old in past.items():
        if name not in now or match_list_regexp(ignore, name):
            continue
        new = now[name]
        read = (new.read_bytes - old.read_bytes) / dt
        write = (new.write_bytes - old.write_bytes) / dt
        if ignore_zero and read == 0 and write == 0:
            continue
        out.append(NamedRate(name=name, read=read, write=write))
    if config.get("sort_disk_ios", True):
        out.sort(key=lambda x: x.read + x.write, reverse=True)
    return out


def _calc_net_io(past, now, dt: float) -> list[NamedRate]:
    ignore = list(config.get("ignore_nets", []) or [])
    ignore_zero = bool(config.get("ignore_0b_net", False))
    out: list[NamedRate] = []
    for name, old in past.items():
        if name not in now or match_list_regexp(ignore, name):
            continue
        new = now[name]
        sent = (new.bytes_sent - old.bytes_sent) / dt
        recv = (new.bytes_recv - old.bytes_recv) / dt
        if ignore_zero and sent == 0 and recv == 0:
            continue
        out.append(NamedRate(name=name, sent=sent, recv=recv))
    if config.get("sort_nets", True):
        out.sort(key=lambda x: x.sent + x.recv, reverse=True)
    return out


async def _test_sites() -> list[SiteResult]:
    sites = list(config.get("test_sites", []) or [])
    timeout = float(config.get("test_timeout", 5) or 5)
    session = get_client_session()

    async def one(site: dict) -> SiteResult:
        name = str(site.get("name") or site.get("url") or "site")
        url = str(site.get("url") or "")
        if not url:
            return SiteResult(name=name, error="无 URL")
        try:
            start = time.perf_counter()
            async with session.get(url, timeout=aiohttp_timeout(timeout), allow_redirects=True) as resp:
                delay = (time.perf_counter() - start) * 1000
                return SiteResult(name=name, status=resp.status, reason=resp.reason or "", delay=delay)
        except asyncio.TimeoutError:
            return SiteResult(name=name, error="超时")
        except Exception as e:
            return SiteResult(name=name, error=type(e).__name__)

    res = await asyncio.gather(*(one(s) for s in sites if isinstance(s, dict)))
    if config.get("sort_sites", True):
        res = list(res)
        res.sort(key=lambda x: x.delay if x.delay is not None else -1)
    return list(res)


def aiohttp_timeout(seconds: float):
    import aiohttp

    return aiohttp.ClientTimeout(total=seconds)


async def _fetch_avatar(self_id: str):
    if self_id in bot_avatar_cache:
        return bot_avatar_cache[self_id]
    from PIL import Image
    import io

    default_path = Path(__file__).resolve().parents[2] / "data" / "status" / "default_avatar.webp"
    img = None
    try:
        session = get_client_session()
        url = f"https://q1.qlogo.cn/g?b=qq&nk={self_id}&s=640"
        async with session.get(url, timeout=aiohttp_timeout(5)) as resp:
            if resp.status == 200:
                data = await resp.read()
                if data:
                    img = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception:
        logger.debug("拉取 Bot 头像失败 self_id=%s", self_id, exc_info=True)
    if img is None and default_path.exists():
        img = Image.open(default_path).convert("RGBA")
    bot_avatar_cache[self_id] = img
    return img


async def _bot_nick(bot: Bot) -> str:
    key = str(bot.self_id)
    if key in bot_nick_cache:
        return bot_nick_cache[key]
    nick = key
    try:
        info = await bot.get_login_info()
        if isinstance(info, dict):
            nick = str(info.get("nickname") or info.get("user_id") or key)
    except Exception:
        logger.debug("获取登录信息失败 self_id=%s", key, exc_info=True)
    bot_nick_cache[key] = nick
    return nick


async def _ob11_msg_num(bot: Bot) -> tuple[int | None, int | None]:
    if not config.get("ob_v11_use_get_status", True):
        return None, None
    try:
        status = await bot.get_status()
    except Exception:
        return None, None
    if not isinstance(status, dict):
        return None, None
    stat = status.get("stat") if isinstance(status.get("stat"), dict) else None
    if not stat:
        return None, None
    rec = stat.get("message_received") or stat.get("MessageReceived")
    sent = stat.get("message_sent") or stat.get("MessageSent")
    try:
        rec_i = int(rec) if rec is not None else None
    except (TypeError, ValueError):
        rec_i = None
    try:
        sent_i = int(sent) if sent is not None else None
    except (TypeError, ValueError):
        sent_i = None
    return rec_i, sent_i


async def collect_bots(current: Bot | None = None) -> list[BotStatus]:
    now = datetime.now()
    bots = [current] if (config.get("show_current_bot_only", False) and current) else list(get_bots().values())
    if not bots and current:
        bots = [current]
    out: list[BotStatus] = []
    for bot in bots:
        if bot is None:
            continue
        key = str(bot.self_id)
        ensure_counters(key)
        connected = format_uptime(now - bot.connected_at) if bot.connected_at else "未知"
        rec, sent = await _ob11_msg_num(bot)
        if rec is None:
            rec = recv_num.get(key)
        if sent is None:
            sent = send_num.get(key)
        nick, avatar = await asyncio.gather(_bot_nick(bot), _fetch_avatar(key))
        out.append(
            BotStatus(
                self_id=key,
                adapter="OneBot V11",
                nick=nick,
                bot_connected=connected,
                msg_rec="未知" if rec is None else str(rec),
                msg_sent="未知" if sent is None else str(sent),
                avatar=avatar,
            )
        )
    return out


def _safe_disk_io():
    try:
        return psutil.disk_io_counters(perdisk=True) or {}
    except Exception:
        return {}


def _safe_net_io():
    try:
        return psutil.net_io_counters(pernic=True) or {}
    except Exception:
        return {}


def _sample_rates_sync():
    psutil.cpu_percent(interval=None)
    procs = []
    try:
        it = psutil.process_iter()
    except Exception:
        it = []
    for proc in it:
        try:
            # 第一次只是打点，返回值一定是 0；1 秒后再取才有 delta
            proc.cpu_percent(interval=None)
            procs.append(proc)
        except (psutil.Error, OSError):
            continue
    past_disk = _safe_disk_io()
    past_net = _safe_net_io()
    t0 = time.time()
    return procs, past_disk, past_net, t0


def _finish_rates_sync(procs, past_disk, past_net, t0: float):
    dt = max(time.time() - t0, 0.001)
    cpu_percent = psutil.cpu_percent(interval=None)
    try:
        freq = psutil.cpu_freq()
    except Exception:
        freq = None
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu_mem = CpuMem(
        cpu_percent=cpu_percent,
        cpu_count=psutil.cpu_count(logical=False),
        cpu_count_logical=psutil.cpu_count(),
        cpu_freq=format_cpu_freq(getattr(freq, "current", None), getattr(freq, "max", None)),
        cpu_brand=_cpu_brand_sync(),
        ram_percent=mem.percent,
        ram_used=mem.used,
        ram_total=mem.total,
        swap_percent=None if swap.total <= 0 else swap.percent,
        swap_used=swap.used,
        swap_total=swap.total,
    )
    disk_io = _calc_disk_io(past_disk, _safe_disk_io(), dt) if past_disk else []
    net_io = _calc_net_io(past_net, _safe_net_io(), dt) if past_net else []
    procs_status = _process_status_sync(procs)
    disks = _disk_usage_sync()
    return cpu_mem, disks, disk_io, net_io, procs_status


async def collect_all(current: Bot | None = None) -> StatusData:
    now = datetime.now()
    primed = await run_in_pool(_sample_rates_sync)
    sites_task = asyncio.create_task(_test_sites())
    bots_task = asyncio.create_task(collect_bots(current))
    await asyncio.sleep(1)
    cpu_mem, disks, disk_io, net_io, procs = await run_in_pool(_finish_rates_sync, *primed)
    sites = await sites_task
    bots = await bots_task
    return StatusData(
        bots=bots,
        bot_run_time=format_uptime(now - started_at),
        system_run_time=format_uptime(now - datetime.fromtimestamp(psutil.boot_time())),
        cpu_mem=cpu_mem,
        disk_usage=disks,
        disk_io=disk_io,
        network_io=net_io,
        network_connection=sites,
        process_status=procs,
        time=now.strftime("%Y-%m-%d %H:%M:%S"),
        python_version=f"{platform.python_implementation()} {platform.python_version()}",
        system_name=_system_name_sync(),
    )