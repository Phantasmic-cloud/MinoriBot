import asyncio
from datetime import datetime, timedelta

from src.utils import *
from src.status import get_status_image_cq

config = Config("alive")
logger = get_logger("alive")
file_db = get_file_db("data/alive/db.json")
cd = ColdDown(file_db, logger)


# ======================= 逻辑处理 ======================= #


def _report_groups() -> list[int]:
    """从配置读取掉线恢复时要通知的群。"""
    groups = config.get("report_groups", []) or []
    out: list[int] = []
    for item in groups:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


@on_bot_connect
async def _notify_connect(bot: Bot) -> None:
    """账号重连后向 report_groups 发一条通知。"""
    groups = _report_groups()
    if not groups:
        return
    msg = f"{bot.self_id} 恢复连接"
    for group_id in groups:
        try:
            await send_group_msg_by_bot(group_id, msg, bot=bot)
        except Exception:
            logger.print_exc(f"向群 {group_id} 发送恢复连接通知失败")


# ======================= 指令处理 ======================= #

alive = CmdHandler(["/alive"], logger)
alive.check_cdrate(cd)


@alive.handle()
async def _(ctx: HandlerContext):
    connected_at = getattr(ctx.bot, "connected_at", None)
    assert_and_reply(connected_at, "当前账号尚未记录连接时间")
    elapsed = datetime.now() - connected_at
    if elapsed < timedelta(0):
        elapsed = timedelta(0)
    await ctx.asend_reply_msg(
        f"当前账号连接时长: {get_readable_timedelta(elapsed, precision='s')}\n"
        f"连接时间: {connected_at.strftime('%Y-%m-%d %H:%M:%S')}"
    )


killbot = CmdHandler(["/killbot"], logger)
killbot.check_superuser()


@killbot.handle()
async def _(ctx: HandlerContext):
    await ctx.asend_reply_msg("正在关闭Bot...")
    await asyncio.sleep(1)
    request_stop()


status = CmdHandler(["状态", "status", "/状态", "/status"], logger, only_to_me=True, block=True)
status.check_cdrate(cd)


@status.handle()
async def _(ctx: HandlerContext):
    return await ctx.asend_msg(await get_status_image_cq(ctx.bot))


# ======================= 定时任务 ======================= #

status_notify_gwl = get_group_white_list(file_db, logger, "status_notify", is_service=False)
STATUS_NOTIFY_TIME = config.get("status_notify_time")


@scheduled_job(
    "cron",
    hour=int(STATUS_NOTIFY_TIME[0]),
    minute=int(STATUS_NOTIFY_TIME[1]),
    second=int(STATUS_NOTIFY_TIME[2]),
)
async def status_notify():
    groups = status_notify_gwl.get()
    if not groups:
        return
    try:
        msg = await get_status_image_cq()
    except Exception:
        logger.print_exc("生成定时状态图失败")
        return
    for group_id in groups:
        try:
            await send_group_msg_by_bot(group_id, msg)
        except Exception:
            logger.print_exc(f"向群 {group_id} 定时推送状态图失败")
