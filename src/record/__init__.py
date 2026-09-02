import asyncio
from datetime import datetime

from src.utils import *
from src.utils.handler import _cmd_history
from .sql import *

config = Config("record")
logger = get_logger("record")
file_db = get_file_db("data/record/db.json")
gbl = get_group_black_list(file_db, logger, "record")
cd = ColdDown(file_db, logger)
record_msg_gbl = get_group_black_list(file_db, logger, "record msg", is_service=False)

message_id_set: set[int] = set()
before_record_hook_funcs = []
after_record_hook_funcs = []
msgs_to_insert: list[dict] = []


# ======================= 逻辑处理 ======================= #


def before_record_hook(func):
    """注册记消息之前要跑的 hook。"""
    before_record_hook_funcs.append(func)
    return func


def after_record_hook(func):
    """注册记消息之后要跑的 hook。"""
    after_record_hook_funcs.append(func)
    return func


async def record_message(bot: Bot, event: MessageEvent):
    """把群消息先过 hook，再攒进待写入队列。"""
    if event.message_id in message_id_set:
        return
    if not is_group_msg(event) and event.user_id == event.self_id:
        return

    message_id_set.add(event.message_id)

    before_hook_tasks = []
    for hook in before_record_hook_funcs:
        async def run_before_hook(hook=hook):
            try:
                await hook(bot, event)
            except Exception:
                logger.print_exc(f"记录消息前hook {hook.__name__} 执行失败")

        before_hook_tasks.append(run_before_hook())
    if before_hook_tasks:
        await asyncio.gather(*before_hook_tasks)

    if record_msg_gbl.check(event, allow_super=False):
        msgs_to_insert.append(
            dict(
                group_id=event.group_id if is_group_msg(event) else 0,
                time=datetime.fromtimestamp(event.time),
                msg_id=event.message_id,
                user_id=event.user_id,
                nickname=get_user_name_by_event(event),
                msg=get_msg(event),
            )
        )

    for hook in after_record_hook_funcs:
        async def run_after_hook(hook=hook):
            try:
                await hook(bot, event)
            except Exception:
                logger.print_exc(f"记录消息后hook {hook.__name__} 执行失败")

        asyncio.create_task(run_after_hook())


# ======================= 定时任务 ======================= #


@repeat_with_interval(config.item("insert_msg_loop_interval_seconds"), "插入消息到数据库", logger)
async def insert_msg_task():
    """把攒着的消息批量写进数据库。"""
    if not msgs_to_insert:
        return
    batch = msgs_to_insert[:]
    msgs_to_insert.clear()
    try:
        await insert_msgs(batch)
    except Exception:
        logger.print_exc(f"插入 {len(batch)} 条消息到数据库失败")


@on_message(priority=-1, block=False)
async def _(bot: Bot, event: MessageEvent):
    """所有消息都走一遍记录。"""
    if not gbl.check(event, allow_private=True, allow_super=False):
        return
    await record_message(bot, event)


# ======================= 指令处理 ======================= #

check = CmdHandler(["/check"], logger)
check.check_superuser()


@check.handle()
async def _(ctx: HandlerContext):
    reply_msg = await ctx.aget_reply_msg()
    assert_and_reply(reply_msg, "请回复一条消息")
    reply = ctx.event.get("reply") if ctx.event else None
    await ctx.asend_reply_msg(str(reply if reply is not None else reply_msg))


nickname = CmdHandler(["/nickname"], logger)
nickname.check_wblist(gbl).check_cdrate(cd)


@nickname.handle()
async def _(ctx: HandlerContext):
    user_id = None
    try:
        cqs = extract_cq_code(ctx.get_msg())
        if "at" in cqs:
            user_id = int(cqs["at"][0]["qq"])
        else:
            user_id = int(ctx.get_args())
    except Exception:
        user_id = ctx.user_id
    assert_and_reply(user_id, "请回复用户或指定用户的QQ号")

    recs = await query_msg_by_user_id(ctx.group_id, user_id)
    recs = sorted(recs, key=lambda x: x["time"])
    if not recs:
        return await ctx.asend_reply_msg(f"用户{user_id}在群{ctx.group_id}中没有发过言")

    nicknames = []
    cur_name = None
    for rec in recs:
        name = rec["nickname"]
        time = rec["time"].strftime("%Y-%m-%d")
        if name != cur_name:
            cur_name = name
            nicknames.append((time, name))

    msg = f"{user_id} 用过的群名片:\n"
    for time, name in nicknames[-50:]:
        msg += f"({time}) {name}\n"
    return await ctx.asend_fold_msg_adaptive(msg.strip())


private_forward = CmdHandler(["/forward"], logger)
private_forward.check_private().check_superuser()


@private_forward.handle()
async def _(ctx: HandlerContext):
    private_forward_list = list(file_db.get("private_forward_list", []) or [])
    user_id = ctx.user_id
    if user_id in private_forward_list:
        private_forward_list.remove(user_id)
        file_db.set("private_forward_list", private_forward_list)
        return await ctx.asend_reply_msg("私聊转发已关闭")
    private_forward_list.append(user_id)
    file_db.set("private_forward_list", private_forward_list)
    return await ctx.asend_reply_msg("私聊转发已开启")


@before_record_hook
async def private_forward_hook(bot: Bot, event: MessageEvent):
    if is_group_msg(event):
        return
    user_id = event.sender.user_id
    nickname = event.sender.nickname
    msg = get_msg(event)
    for forward_user_id in file_db.get("private_forward_list", []) or []:
        if user_id == forward_user_id:
            continue
        await send_private_msg_by_bot(forward_user_id, f"来自{nickname}({user_id})的私聊消息:", bot=bot)
        await send_private_msg_by_bot(forward_user_id, msg, bot=bot)


get_cmd_history = CmdHandler(["/cmd_history", "/cmdh"], logger)
get_cmd_history.check_superuser()


@get_cmd_history.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args()
    try:
        limit = int(args)
    except Exception:
        limit = 10
    if limit <= 0:
        limit = 10
    history = _cmd_history[-limit:]
    msg = "【历史记录】\n"
    for context in history:
        time = context.time.strftime("%Y-%m-%d %H:%M:%S") if context.time else "-"
        msg += f"[{time}]\n"
        group_id, user_id = context.group_id, context.user_id
        if group_id:
            group_name = await get_group_name(ctx.bot, group_id)
            msg += f"<{group_name}({group_id})>\n"
            user_name = await get_group_member_name(group_id, user_id, ctx.bot)
            msg += f"<{user_name}({user_id})>\n"
        else:
            sender = getattr(context.event, "sender", None) if context.event else None
            user_name = getattr(sender, "nickname", None) or str(user_id)
            msg += f"<{user_name}({user_id})>\n"
        msg += f"{context.trigger_cmd} {context.arg_text}"
        msg += "\n\n"
    return await ctx.asend_fold_msg_adaptive(msg.strip())


forward_to_text = CmdHandler(["/转文本", "/to_text", "/to text"], logger)
forward_to_text.check_wblist(gbl).check_cdrate(cd)


@forward_to_text.handle()
async def _(ctx: HandlerContext):
    def json_msg_to_readable_text(mdata: dict):
        data = None
        try:
            raw = mdata.get("data")
            data = loads_json(raw) if isinstance(raw, (str, bytes)) else raw
            title = data["meta"]["detail_1"]["title"]
            desc = truncate(data["meta"]["detail_1"]["desc"], 32)
            return f"[{title}分享:{desc}]"
        except Exception:
            try:
                return f"[转发消息:{data['prompt']}]"
            except Exception:
                return "[转发消息(加载失败)]"

    async def get_forward_msg_text(bot, forward_seg, indent: int = 0) -> str:
        forward_id = forward_seg["data"]["id"]
        forward_content = forward_seg["data"].get("content")
        if not forward_content:
            forward_msg = await get_forward_msg(bot, forward_id)
            if not forward_msg:
                return "[转发消息(加载失败)]"
            forward_content = forward_msg.get("messages") or []

        text = " " * indent + "=== 折叠消息 ===\n"
        for msg_obj in forward_content:
            sender = msg_obj.get("sender") or {}
            sender_name = sender.get("nickname") or str(sender.get("user_id") or "")
            segs = msg_obj.get("message") or []
            text += " " * indent + f"{sender_name}: "
            for seg in segs:
                mtype, mdata = seg.get("type"), seg.get("data") or {}
                if mtype == "text":
                    text += f"{mdata.get('text', '')}"
                elif mtype == "face":
                    text += "[表情]"
                elif mtype == "image":
                    text += "[图片]" if str(mdata.get("sub_type", 0) or 0) in ("0", "") else "[表情]"
                elif mtype == "video":
                    text += "[视频]"
                elif mtype == "audio":
                    text += "[音频]"
                elif mtype == "file":
                    text += "[文件]"
                elif mtype == "at":
                    text += f"[@{mdata.get('qq')}]"
                elif mtype == "reply":
                    text += f"[reply={mdata.get('id')}]"
                elif mtype == "forward":
                    text += await get_forward_msg_text(bot, seg, indent + 4)
                elif mtype == "json":
                    text += json_msg_to_readable_text(mdata)
            text += "\n"
        text += " " * indent + "============\n"
        return text

    reply_msg = await ctx.aget_reply_msg()
    assert_and_reply(reply_msg, "请回复一条聊天记录")
    reply_msg = Message.of(reply_msg).to_rich()
    forward_seg = None
    for seg in reply_msg:
        if seg.get("type") == "forward":
            forward_seg = seg
            break
    assert_and_reply(forward_seg, "回复的消息不是聊天记录")
    text = await get_forward_msg_text(ctx.bot, forward_seg)
    return await ctx.asend_fold_msg_adaptive(text)
