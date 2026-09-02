import asyncio
import base64
import glob
import hashlib
import io
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Callable
from uuid import uuid4

import aiohttp
from tenacity import retry, stop_after_attempt, wait_fixed

from src.core import Config, get_cfg_or_value, get_logger

utils_logger = get_logger("utils")


def _thread_pool_size() -> int:
    n = int(Config("global").get("default_thread_pool_size"))
    return max(1, n)


_pool = ThreadPoolExecutor(max_workers=_thread_pool_size())
_session: aiohttp.ClientSession | None = None


def get_exc_desc(e: Exception) -> str:
    et = type(e).__name__
    if et in ("Exception", "AssertionError", "ReplyException"):
        et = ""
    msg = str(e)
    if et and msg:
        return f"{et}: {msg}"
    return et + msg


def truncate(s: Any, limit: int) -> str:
    if s is None:
        return "<None>"
    s = str(s)
    length = 0
    for i, ch in enumerate(s):
        if length >= limit:
            return s[:i] + "..."
        length += 1 if ord(ch) < 128 else 2
    return s


def get_str_display_length(s: str) -> int:
    length = 0
    for ch in str(s):
        length += 1 if ord(ch) < 128 else 2
    return length


def get_str_line_count(s: str, line_length: int) -> int:
    lines = [""]
    for c in str(s):
        if c == "\n":
            lines.append("")
            continue
        if get_str_display_length(lines[-1] + c) > line_length:
            lines.append("")
        lines[-1] += c
    return len(lines)


def get_readable_file_size(size: int) -> str:
    size = int(size or 0)
    if size < 1024:
        return f"{size}B"
    size /= 1024
    if size < 1024:
        return f"{size:.2f}KB"
    size /= 1024
    if size < 1024:
        return f"{size:.2f}MB"
    size /= 1024
    return f"{size:.2f}GB"


def get_readable_timedelta(delta: timedelta, precision: str = "m") -> str:
    match precision:
        case "s":
            keep = 3
        case "m":
            keep = 2
        case "h":
            keep = 1
        case "d":
            keep = 0
        case _:
            keep = 2
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "0秒"
    days, seconds = divmod(seconds, 24 * 3600)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}天")
    if hours and (keep >= 1 or not parts):
        parts.append(f"{hours}小时")
    if minutes and (keep >= 2 or not parts):
        parts.append(f"{minutes}分钟")
    if seconds and (keep >= 3 or not parts):
        parts.append(f"{seconds}秒")
    return "".join(parts) or "0秒"


def get_md5(s: str | bytes) -> str:
    if isinstance(s, str):
        s = s.encode()
    return hashlib.md5(s).hexdigest()


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]


def find_by(lst: list[dict[str, Any]], key: str, value: Any, mode: str = "first"):
    matched = [item for item in lst if key in item and item[key] == value]
    if mode == "all":
        return matched
    if not matched:
        return None
    return matched[0] if mode == "first" else matched[-1]


def loads_json(s: str | bytes) -> Any:
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return json.loads(s)


def dumps_json(data: Any, indent: bool = True) -> str:
    if indent:
        return json.dumps(data, ensure_ascii=False, indent=2)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def create_parent_folder(file_path) -> str:
    file_path = str(file_path)
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return file_path


def remove_folder(folder_path) -> None:
    folder_path = str(folder_path)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def get_client_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def close_client_session() -> None:
    global _session
    if _session is None or _session.closed:
        _session = None
        return
    await _session.close()
    _session = None


def shutdown_thread_pool() -> None:
    try:
        _pool.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        _pool.shutdown(wait=False)


async def run_in_pool(func: Callable, *args, pool=None):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(pool or _pool, func, *args)


async def batch_gather(*futs_or_coros, batch_size=32) -> list[Any]:
    results = []
    for i in range(0, len(futs_or_coros), batch_size):
        results.extend(await asyncio.gather(*futs_or_coros[i : i + batch_size]))
    return results


async def call_common_or_async(func: Callable, *args, **kwargs):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    ret = func(*args, **kwargs)
    if asyncio.iscoroutine(ret):
        return await ret
    return ret


def _startup_delay() -> float:
    cfg = Config("global")
    lo = float(cfg.get("startup_task_delay_seconds.min"))
    hi = float(cfg.get("startup_task_delay_seconds.max"))
    if hi < lo:
        hi = lo
    return random.uniform(lo, hi)


_pending_startup_tasks: list[Callable] = []


def start_repeat_with_interval(
    interval,
    func: Callable,
    logger,
    name: str,
    every_output: bool = False,
    error_output: bool = True,
    error_limit: int = 5,
    delay: float | None = None,
) -> None:
    """开始重复执行某个任务，启动时打一行「开始循环执行 {name} 任务」。"""
    if delay is None:
        delay = _startup_delay()

    async def task():
        await asyncio.sleep(delay)
        try:
            error_count = 0
            logger.info("开始循环执行 %s 任务", name)
            next_time = datetime.now() + timedelta(seconds=1)
            while True:
                now_time = datetime.now()
                if next_time > now_time:
                    try:
                        await asyncio.sleep((next_time - now_time).total_seconds())
                    except asyncio.CancelledError:
                        return
                    except Exception:
                        logger.print_exc(f"循环执行 {name} sleep失败")
                next_time = next_time + timedelta(seconds=float(get_cfg_or_value(interval, 60) or 60))
                try:
                    if every_output:
                        logger.debug("开始执行 %s", name)
                    await call_common_or_async(func)
                    if every_output:
                        logger.info("执行 %s 成功", name)
                    if error_output and error_count > 0:
                        logger.info("循环执行 %s 从错误中恢复, 累计错误次数: %s", name, error_count)
                    error_count = 0
                except Exception as e:
                    if error_output and error_count < error_limit - 1:
                        logger.warning("循环执行 %s 失败: %s (失败次数 %s)", name, e, error_count + 1)
                    elif error_output and error_count == error_limit - 1:
                        logger.print_exc(f"循环执行 {name} 失败 (达到错误次数输出上限)")
                    error_count += 1
        except Exception:
            logger.print_exc(f"循环执行 {name} 任务失败")

    _pending_startup_tasks.append(task)
    try:
        from apscheduler.schedulers.base import STATE_RUNNING

        from src.utils.scheduler import scheduler

        if scheduler.state == STATE_RUNNING:
            asyncio.get_running_loop().create_task(task())
            _pending_startup_tasks.remove(task)
    except RuntimeError:
        pass


def repeat_with_interval(
    interval_secs,
    name: str,
    logger,
    every_output: bool = False,
    error_output: bool = True,
    error_limit: int = 5,
    delay: float | None = None,
):
    """重复执行某个任务的装饰器。"""

    def wrapper(func):
        start_repeat_with_interval(
            interval_secs, func, logger, name, every_output, error_output, error_limit, delay
        )
        return func

    return wrapper


def start_async_task(func: Callable, logger, name: str, delay=None):
    """开始异步执行某个任务。"""
    if delay is None:
        delay = _startup_delay()

    async def task():
        await asyncio.sleep(delay)
        try:
            logger.info("开始异步执行 %s 任务", name)
            await call_common_or_async(func)
        except Exception:
            logger.print_exc(f"异步执行 {name} 任务失败")

    _pending_startup_tasks.append(task)
    try:
        from apscheduler.schedulers.base import STATE_RUNNING

        from src.utils.scheduler import scheduler

        if scheduler.state == STATE_RUNNING:
            asyncio.get_running_loop().create_task(task())
            _pending_startup_tasks.remove(task)
    except RuntimeError:
        pass


def async_task(name: str, logger, delay=None):
    """异步执行某个任务的装饰器。"""

    def wrapper(func):
        start_async_task(func, logger, name, delay)
        return func

    return wrapper


def start_pending_startup_tasks() -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    tasks = list(_pending_startup_tasks)
    _pending_startup_tasks.clear()
    for task in tasks:
        loop.create_task(task())


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
async def download_file(url, file_path):
    async with get_client_session().get(url, ssl=False) as resp:
        if resp.status != 200:
            raise Exception(f"下载文件 {truncate(url, 32)} 失败: {resp.status} {resp.reason}")
        with open(file_path, "wb") as f:
            f.write(await resp.read())


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
async def download_image(image_url, force_http=False):
    from PIL import Image

    if force_http and str(image_url).startswith("https"):
        image_url = str(image_url).replace("https", "http", 1)
    async with get_client_session().get(image_url, ssl=False) as resp:
        if resp.status != 200:
            raise Exception(f"下载图片失败: {resp.status} {resp.reason}")
        image = Image.open(io.BytesIO(await resp.read()))
        image.load()
        return image


def get_image_b64(img) -> str:
    from PIL import Image

    if isinstance(img, str):
        return img
    if not isinstance(img, Image.Image):
        raise TypeError(f"不支持的图片类型: {type(img)}")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_to_image(s: str):
    from PIL import Image

    if "," in s:
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s)))


async def download_image_to_b64(image_path) -> str:
    """下载图片并编码为带 data:image 前缀的 base64 字符串。"""
    img = await download_image(image_path)
    return get_image_b64(img)


TEMP_FILE_DIR = "data/utils/tmp"
_tmp_files_to_remove: list[tuple[str, datetime]] = []


class TempFilePath:
    def __init__(self, ext: str, remove_after: timedelta | None = None) -> None:
        if ext.startswith("."):
            ext = ext[1:]
        os.makedirs(TEMP_FILE_DIR, exist_ok=True)
        self.path = os.path.abspath(os.path.join(TEMP_FILE_DIR, f"{uuid4()}.{ext}"))
        self.remove_after = remove_after

    def __enter__(self) -> str:
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.remove_after is None:
            try:
                os.remove(self.path)
            except OSError:
                pass
        else:
            _tmp_files_to_remove.append((self.path, datetime.now() + self.remove_after))


class TempDownloadFilePath(TempFilePath):
    def __init__(self, url, ext: str | None = None, remove_after: timedelta | None = None):
        self.url = url
        if ext is None:
            ext = str(url).split(".")[-1].split("?")[0] or "bin"
        super().__init__(ext, remove_after)

    async def __aenter__(self) -> str:
        await download_file(self.url, self.path)
        return super().__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)


def _remove_path(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        utils_logger.print_exc(f"删除临时文件 {path} 失败")


def clean_tmp_files() -> None:
    """删掉到期的 TempFilePath，以及临时目录里超过一天的文件。"""
    global _tmp_files_to_remove
    now = datetime.now()
    remain: list[tuple[str, datetime]] = []
    for path, remove_time in list(_tmp_files_to_remove):
        if now >= remove_time:
            _remove_path(path)
        else:
            remain.append((path, remove_time))
    _tmp_files_to_remove = remain
    try:
        files = glob.glob(os.path.join(TEMP_FILE_DIR, "*"))
    except Exception:
        return
    for file in files:
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(file))
            if now - mtime > timedelta(days=1):
                _remove_path(file)
        except Exception:
            utils_logger.print_exc(f"删除临时文件 {file} 失败")


@repeat_with_interval(60, "清除临时文件", utils_logger, delay=1)
def _clean_tmp_files_loop() -> None:
    clean_tmp_files()


async def get_image_cq(
    image,
    allow_error: bool = False,
    logger=None,
    low_quality: bool = False,
    quality=None,
    subsampling=None,
    optimize=None,
    send_url_as_is: bool = False,
):
    """把图片转成 CQ 码。支持路径、URL、bytes、PIL.Image。"""
    from PIL import Image

    global_cfg = Config("global")
    if quality is None:
        quality = global_cfg.get("msg_send.low_quality_image.default_quality")
    if subsampling is None:
        subsampling = global_cfg.get("msg_send.low_quality_image.default_subsampling")
    if optimize is None:
        optimize = global_cfg.get("msg_send.low_quality_image.default_optimize")
    keep_minutes = int(global_cfg.get("msg_send.tmp_img_keep_minutes"))

    try:
        if isinstance(image, str) and image.startswith("http"):
            if send_url_as_is:
                return f"[CQ:image,file={image}]"
            session = get_client_session()
            async with session.get(image) as resp:
                resp.raise_for_status()
                image = Image.open(io.BytesIO(await resp.read()))
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"图片文件不存在: {image}")
            if send_url_as_is:
                return f"[CQ:image,file=file://{os.path.abspath(image)}]"
            image = Image.open(image)
            image.load()

        is_gif = bool(getattr(image, "is_animated", False) or image.mode == "P")
        ext = "gif" if is_gif else ("jpg" if low_quality else "png")
        with TempFilePath(ext, remove_after=timedelta(minutes=keep_minutes)) as tmp_path:
            if ext == "gif":
                from src.draw.img_utils import get_gif_duration, gif_to_frames, save_transparent_gif

                save_transparent_gif(gif_to_frames(image), get_gif_duration(image), tmp_path)
            elif ext == "jpg":
                image.convert("RGB").save(
                    tmp_path,
                    format="JPEG",
                    quality=int(get_cfg_or_value(quality, 75)),
                    optimize=bool(get_cfg_or_value(optimize, True)),
                    subsampling=int(get_cfg_or_value(subsampling, 2)),
                    progressive=False,
                )
            else:
                image.save(tmp_path)
            return f"[CQ:image,file=file://{os.path.abspath(tmp_path)}]"
    except Exception as e:
        if allow_error:
            if logger is not None:
                logger.print_exc(f"生成图片CQ码失败: {e}")
            return "[图片发送失败]"
        raise
