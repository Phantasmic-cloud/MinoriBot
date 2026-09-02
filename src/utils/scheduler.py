from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.base import STATE_RUNNING

from src.core import get_logger

logger = get_logger("utils")
TZ = ZoneInfo("Asia/Shanghai")

scheduler = AsyncIOScheduler(timezone=TZ)


def start_scheduler() -> None:
    """启动定时任务调度器，并跑启动时积压的任务。"""
    if scheduler.state == STATE_RUNNING:
        return
    scheduler.start()
    logger.info("定时任务调度器已启动 timezone=%s", TZ)
    from src.utils.utils import start_pending_startup_tasks

    start_pending_startup_tasks()


def stop_scheduler(wait: bool = False) -> None:
    """关掉定时任务调度器。"""
    if scheduler.state != STATE_RUNNING:
        return
    scheduler.shutdown(wait=wait)
    logger.info("定时任务调度器已停止")


def add_job(func: Callable, trigger: str | None = None, **kwargs: Any):
    """注册任务。trigger 支持 cron / interval / date，参数和 APScheduler 一样。"""
    return scheduler.add_job(func, trigger, **kwargs)


def scheduled_job(trigger: str, **kwargs: Any):
    """装饰器版，插件里直接 @scheduled_job('cron', hour=0, minute=0)。"""
    return scheduler.scheduled_job(trigger, **kwargs)


def remove_job(job_id: str) -> None:
    scheduler.remove_job(job_id)


def get_job(job_id: str):
    return scheduler.get_job(job_id)


def now() -> datetime:
    return datetime.now(TZ)


__all__ = [
    "TZ",
    "add_job",
    "get_job",
    "now",
    "remove_job",
    "scheduled_job",
    "scheduler",
    "start_scheduler",
    "stop_scheduler",
]
