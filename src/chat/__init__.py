import copy
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import openai

from src.code.run import run as run_code
from src.llm import ChatSession, api_provider_mgr, get_model_preset, translate_text, tts
from src.utils import *
from .autochat import *
from .sticker import *

config = Config("chat.chat")
logger = get_logger("chat")
file_db = get_file_db("data/chat/db.json")
gwl = get_group_white_list(file_db, logger, "chat")
at_trigger_chat_gbl = get_group_black_list(file_db, logger, "atchat", is_service=False)

chat_cd = ColdDown(file_db, logger, config.item("chat_cd"), cold_down_name="chat_cd")
tts_cd = ColdDown(file_db, logger, config.item("tts_cd"), cold_down_name="tts_cd")
img_trans_cd = ColdDown(file_db, logger, config.item("img_trans_cd"), cold_down_name="img_trans_cd")

SESSION_LEN_LIMIT_CFG = config.item("session_len_limit")
IMAGE_CAPTION_LIMIT_CFG = config.item("image_caption.limit")
IMAGE_CAPTION_TIMEOUT_SEC_CFG = config.item("image_caption.timeout_sec")

SYSTEM_PROMPT_PATH = Path("config/chat/system_prompt.txt")
SYSTEM_PROMPT_TOOLS_PATH = Path("config/chat/system_prompt_tools.txt")
TOOLS_TRIGGER_WORDS_PATH = Path("config/chat/tools_trigger_words.txt")
SYSTEM_PROMPT_PYTHON_RET = Path("config/chat/system_prompt_python_ret.txt")
IMAGE_CAPTION_TEMPLATE_PATH = Path("config/chat/image_caption_prompt.txt")

CLEANCHAT_TRIGGER_WORDS = ["cleanchat", "clean_chat", "cleanmode", "clean_mode"]
IMAGE_RESPONSE_TRIGGER_WORDS = ["生成图片", "图片生成", "imagen", "Imagen", "IMAGEN"]

image_caption_db = get_file_db("data/chat/image_caption_db.json")
SESSION_EXPIRE_TIME = timedelta(hours=12)
sessions: dict[str, ChatSession] = {}
query_msg_ids = set()


# ======================= 逻辑处理 ======================= #


def _reply_sender_id(sender) -> int | None:
    """从回复消息的 sender 里取出 QQ。"""

    if sender is None:
        return None
    if isinstance(sender, dict):
        uid = sender.get("user_id")
    else:
        uid = getattr(sender, "user_id", None)
    try:
        return int(uid)
    except (TypeError, ValueError):
        return None

async def use_tool(ctx: HandlerContext, session: ChatSession, type: str, data: Any) -> str:
    """执行聊天会话里请求的工具，目前只支持 python。"""

    if type == "python":
        logger.info("使用python工具, data: %s", data)
        await ctx.asend_fold_msg_adaptive(f"正在执行python代码:\n\n{data}")
        try:
            res = await run_code("py\n" + data)
        except Exception as e:
            logger.print_exc("请求运行代码失败")
            res = f"运行代码失败: {get_exc_desc(e)}"
        logger.info("python执行结果: %s", res)
        system_prompt_ret = SYSTEM_PROMPT_PYTHON_RET.read_text(encoding="utf-8")
        session.append_system_content(system_prompt_ret.format(res=res))
        return res
    raise Exception("unknown tool type")

async def get_image_caption(mdata: dict, model_name, timeout: int, use_llm: bool):
    """用 LLM 总结图片内容，结果按 file_unique 缓存。"""

    summary = mdata.get("summary", "")
    url = mdata.get("url", None)
    file_unique = mdata.get("file_unique", "")
    sub_type = mdata.get("sub_type", mdata.get("subType", 0))
    try:
        sub_type = int(sub_type or 0)
    except (TypeError, ValueError):
        sub_type = 0
    sub_type = "图片" if sub_type == 0 else "表情"
    caption = image_caption_db.get(file_unique) if file_unique else None
    if not caption:
        logger.info(
            "chat尝试总结图片: file_unique=%s url=%s summary=%s subtype=%s",
            file_unique, url, summary, sub_type,
        )
        try:
            if not use_llm or not url:
                return f"[{sub_type}(加载失败)]" if not summary else f"[{sub_type}:{summary}]"

            prompt = IMAGE_CAPTION_TEMPLATE_PATH.read_text(encoding="utf-8").format(sub_type=sub_type)
            img = await download_image_to_b64(url)
            session = ChatSession()
            session.append_user_content(prompt, imgs=[img], verbose=False)
            resp = await session.get_response(model_name=model_name, timeout=timeout)
            caption = truncate(resp.result.strip(), 512)
            assert caption, "图片总结为空"

            logger.info("图片总结成功: %s", caption)
            if file_unique:
                image_caption_db.set(file_unique, caption)
                keys = image_caption_db.get("keys", [])
                keys.append(file_unique)
                while len(keys) > IMAGE_CAPTION_LIMIT_CFG.get():
                    key = keys.pop(0)
                    image_caption_db.delete(key)
                    logger.info("删除图片caption: %s", key)
                image_caption_db.set("keys", keys)
        except Exception:
            logger.print_exc(f"总结图片 url={url} 失败")
            return f"[{sub_type}(加载失败)]" if not summary else f"[{sub_type}:{summary}]"
    return f"[{sub_type}:{caption}]"

def json_msg_to_readable_text(mdata: dict):
    """把 json 卡片转成可读文本。"""

    try:
        data = loads_json(mdata["data"])
        title = data["meta"]["detail_1"]["title"]
        desc = truncate(data["meta"]["detail_1"]["desc"], 32)
        return f"[{title}分享:{desc}]"
    except Exception:
        try:
            data = loads_json(mdata["data"])
            return f"[转发消息:{data['prompt']}]"
        except Exception:
            return "[转发消息(加载失败)]"

async def get_forward_msg_text(bot: Bot, model, forward_seg, indent: int = 0) -> str:
    """把转发聊天记录展成文本，图片会顺带总结。"""

    logger.info("chat开始总结聊天记录: %s", forward_seg["data"].get("id"))
    forward_id = forward_seg["data"].get("id")
    forward_content = forward_seg["data"].get("content")
    if not forward_content:
        forward_msg = await get_forward_msg(bot, forward_id)
        if not forward_msg:
            logger.warning("chat获取聊天记录失败: %s", forward_id)
            return "[转发消息(加载失败)]"
        forward_content = forward_msg["messages"]

    text = " " * indent + "聊天记录```\n"
    for msg_obj in forward_content:
        sender = msg_obj.get("sender") or {}
        sender_name = sender.get("nickname") or sender.get("card") or str(sender.get("user_id") or "")
        segs = msg_obj.get("message") or []
        text += " " * indent + f"{sender_name}: "
        for seg in segs:
            mtype, mdata = seg.get("type"), seg.get("data") or {}
            if mtype == "text":
                text += f"{mdata.get('text', '')}"
            elif mtype == "face":
                text += "[表情]"
            elif mtype == "image":
                use_llm = int(mdata.get("sub_type", mdata.get("subType", 0)) or 0) == 0
                text += await get_image_caption(mdata, model, IMAGE_CAPTION_TIMEOUT_SEC_CFG.get(), use_llm=use_llm)
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
                text += await get_forward_msg_text(bot, model, seg, indent + 4)
            elif mtype == "json":
                text += json_msg_to_readable_text(mdata)
        text += "\n"
    text += " " * indent + "```\n"
    return text

def trigger_chat_help_condition(text: str) -> bool:
    """判断是否该弹出 /chat 帮助。"""

    if "/chat" not in text:
        return False
    text = text.strip().replace("/chat", "")
    return text in ["help", "帮助"]

def get_group_model_name(group_id, mode):
    """取群聊当前 mode 用的模型名。"""

    group_model_dict = file_db.get("group_chat_model_dict", {})
    default = get_model_preset("chat.group")
    return group_model_dict.get(str(group_id), default).get(mode, default[mode])

def get_private_model_name(user_id, mode):
    """取私聊当前 mode 用的模型名。"""

    private_model_dict = file_db.get("private_chat_model_dict", {})
    default = get_model_preset("chat.private")
    return private_model_dict.get(str(user_id), default).get(mode, default[mode])

def get_model_name(event, mode):
    """按群聊/私聊取当前 mode 用的模型名。"""

    if is_group_msg(event):
        ret = get_group_model_name(event.group_id, mode)
    else:
        ret = get_private_model_name(event.user_id, mode)
    if not isinstance(ret, str) and len(ret) == 1:
        ret = ret[0]
    return ret

def clear_group_model_name(group_id):
    """清空群聊的自定义模型。"""

    group_model_dict = file_db.get("group_chat_model_dict", {})
    group_model_dict.pop(str(group_id), None)
    file_db.set("group_chat_model_dict", group_model_dict)

def clear_private_model_name(user_id):
    """清空私聊的自定义模型。"""

    private_model_dict = file_db.get("private_chat_model_dict", {})
    private_model_dict.pop(str(user_id), None)
    file_db.set("private_chat_model_dict", private_model_dict)

def clear_model_name(event):
    """清空当前会话的自定义模型。"""

    if is_group_msg(event):
        clear_group_model_name(event.group_id)
    else:
        clear_private_model_name(event.user_id)

def change_group_model_name(group_id, model_name: str, mode):
    """切换群聊某个 mode 的模型。"""

    ChatSession.check_model_name(model_name, mode)
    group_model_dict = file_db.get("group_chat_model_dict", {})
    default = get_model_preset("chat.group")
    if str(group_id) not in group_model_dict:
        group_model_dict[str(group_id)] = copy.deepcopy(default)
    group_model_dict[str(group_id)][mode] = model_name
    file_db.set("group_chat_model_dict", group_model_dict)

def change_private_model_name(user_id, model_name: str, mode):
    """切换私聊某个 mode 的模型。"""

    ChatSession.check_model_name(model_name, mode)
    private_model_dict = file_db.get("private_chat_model_dict", {})
    default = get_model_preset("chat.private")
    if str(user_id) not in private_model_dict:
        private_model_dict[str(user_id)] = copy.deepcopy(default)
    private_model_dict[str(user_id)][mode] = model_name
    file_db.set("private_chat_model_dict", private_model_dict)

def change_model_name(event, model_name: str, mode):
    """切换当前会话某个 mode 的模型，返回规范化后的全名。"""

    model_name = api_provider_mgr.find_model(model_name).get_full_name()
    if is_group_msg(event):
        change_group_model_name(event.group_id, model_name, mode)
    else:
        change_private_model_name(event.user_id, model_name, mode)
    return model_name


# ======================= 指令处理 ======================= #

CHAT_CMDS = ["/chat"]
chat_request = CmdHandler(
    [""],
    logger,
    block=False,
    help_command="/chat",
    help_trigger_condition=trigger_chat_help_condition,
)


@chat_request.handle()
async def _(ctx: HandlerContext):
    bot, event = ctx.bot, ctx.event
    global sessions, query_msg_ids
    session = None
    session_id_backup = None
    try:
        if not gwl.check(event, allow_private=True, allow_super=True):
            return
        if check_self_reply(event):
            return
        if is_group_msg(event) and not command_text(event) and not event.to_me:
            return

        query_msg = ctx.get_msg()
        query_text = extract_text(query_msg)
        query_imgs = extract_image_url(query_msg)
        query_cqs = extract_cq_code(query_msg)
        reply_msg = await ctx.aget_reply_msg()
        reply_id = ctx.get_reply_msg_id()

        triggered_by_chat_cmd = False
        for chat_cmd in CHAT_CMDS:
            if query_text.strip().startswith(chat_cmd):
                query_text = query_text.strip().removeprefix(chat_cmd)
                triggered_by_chat_cmd = True
                break

        if is_group_msg(event) and (autochat_gwl.check_id(event.group_id) or not at_trigger_chat_gbl.check(event)):
            if not triggered_by_chat_cmd:
                return

        if query_text.strip().startswith("/"):
            return

        if is_group_msg(event):
            bot_name = await get_group_member_name(event.group_id, bot.self_id, bot=bot)
        else:
            try:
                info = await bot.get_login_info()
                bot_name = (info or {}).get("nickname") or str(bot.self_id)
            except Exception:
                bot_name = str(bot.self_id)

        if query_text.replace(f"@{bot_name}", "").strip() == "" or query_text is None:
            return

        has_true_at = False
        has_text_at = False
        if "at" in query_cqs:
            for cq in query_cqs["at"]:
                if str(cq.get("qq")) == str(bot.self_id):
                    has_true_at = True
                    break
        if "text" in query_cqs:
            for cq in query_cqs["text"]:
                if f"@{bot_name}" in str(cq.get("text") or ""):
                    has_text_at = True
                    break
        if not triggered_by_chat_cmd and (is_group_msg(event) or check_self(event)):
            if not (has_true_at or has_text_at):
                return

        if not await chat_cd.check(event):
            return

        logger.info("收到询问: %s", query_msg)
        query_msg_ids.add(event.message_id)
        model_name = None

        if has_text_at:
            query_text = query_text.replace(f"@{bot_name}", "")

        if "model:" in query_text:
            if is_group_msg(event) and not check_superuser(event):
                return await ctx.asend_reply_msg("非超级用户不允许自定义模型")
            model_name = query_text.split("model:")[1].strip().split(" ")[0]
            try:
                ChatSession.check_model_name(model_name)
            except Exception as e:
                return await ctx.asend_reply_msg(f"{e}")
            query_text = query_text.replace(f"model:{model_name}", "").strip()
            logger.info("使用指定模型: %s", model_name)

        if any(word in query_text for word in CLEANCHAT_TRIGGER_WORDS):
            for word in CLEANCHAT_TRIGGER_WORDS:
                query_text = query_text.replace(word, "")
            need_tools = False
            system_prompt = None
            logger.info("使用CleanChat模式")
        else:
            tools_trigger_words = TOOLS_TRIGGER_WORDS_PATH.read_text(encoding="utf-8").split()
            need_tools = any(word and word in query_text for word in tools_trigger_words)
            logger.info("使用工具: %s", need_tools)
            system_prompt_path = SYSTEM_PROMPT_TOOLS_PATH if need_tools else SYSTEM_PROMPT_PATH
            system_prompt = system_prompt_path.read_text(encoding="utf-8").format(
                bot_name=bot_name,
                current_date=datetime.now().strftime("%Y-%m-%d"),
            )

        enable_image_response = False
        if any(word in query_text for word in IMAGE_RESPONSE_TRIGGER_WORDS):
            for word in IMAGE_RESPONSE_TRIGGER_WORDS:
                query_text = query_text.replace(word, "")
            enable_image_response = True
            query_text += "\n生成图片作为回复"
            system_prompt = None
            logger.info("使用生成图片模式")

        if reply_msg is not None:
            logger.info("回复模式：%s", reply_id)
            if str(reply_id) in sessions:
                session = sessions[str(reply_id)]
                sessions.pop(str(reply_id))
                session_id_backup = reply_id
                logger.info("沿用会话%s, 长度:%s", session.id, len(session))
            else:
                reply_text = extract_text(reply_msg)
                reply_cqs = extract_cq_code(reply_msg)
                reply_imgs = extract_image_url(reply_msg)
                reply_uid = _reply_sender_id(ctx.get_reply_sender())
                logger.info("获取回复消息: %s, uid:%s", reply_id, reply_uid)
                if any(t in reply_cqs for t in ["json", "video"]):
                    return
                session = ChatSession(system_prompt)
                if "forward" in reply_cqs:
                    logger.info(reply_cqs["forward"][0].get("id"))
                    forward_seg = {"type": "forward", "data": reply_cqs["forward"][0]}
                    forward_text = await get_forward_msg_text(
                        ctx.bot,
                        get_model_preset("chat.image_caption"),
                        forward_seg,
                    )
                    session.append_user_content(forward_text)
                elif reply_imgs or reply_text.strip() != "":
                    reply_imgs = [await download_image_to_b64(img) for img in reply_imgs]
                    if str(reply_uid) == str(bot.self_id):
                        if reply_imgs:
                            session.append_user_content(reply_text, reply_imgs)
                        else:
                            session.append_bot_content(reply_text)
                    else:
                        session.append_user_content(reply_text, reply_imgs)
        else:
            session = ChatSession(system_prompt)

        query_imgs = [await download_image_to_b64(img) for img in query_imgs]
        session.append_user_content(query_text, query_imgs)
        if len(session) == 0:
            return

        if not model_name:
            mode = "text"
            if enable_image_response:
                mode = "image"
            elif need_tools:
                mode = "tool"
            elif session.has_multimodal_content():
                mode = "mm"
            model_name = get_model_name(event, mode)

        total_seconds, total_ptokens, total_ctokens, total_cost = 0, 0, 0, 0
        tools_additional_info = ""
        rest_quota = 0
        reasoning = None
        resp_model = None
        res_text = ""

        for _ in range(3):
            t = datetime.now()
            resp = await session.get_response(
                model_name=model_name,
                image_response=enable_image_response,
                timeout=300,
            )

            res_text = ""
            for part in resp.result_list:
                if isinstance(part, str):
                    b64_pattern = re.compile(r"!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)")
                    last_end = 0
                    for m in b64_pattern.finditer(part):
                        if m.start() > last_end:
                            res_text += part[last_end:m.start()]
                        try:
                            img = b64_to_image(m.group(1))
                            res_text += await get_image_cq(img)
                        except Exception as e:
                            logger.warning("base64图片转换失败: %s", e)
                        last_end = m.end()
                    res_text += part[last_end:]
                else:
                    res_text += await get_image_cq(part)
            res_text = res_text.strip()

            total_ptokens += resp.prompt_tokens
            total_ctokens += resp.completion_tokens
            total_cost += resp.cost
            total_seconds += (datetime.now() - t).total_seconds()
            rest_quota = resp.quota
            resp_model = resp.model
            reasoning = resp.reasoning

            if not gwl.check(event, allow_private=True, allow_super=True):
                return
            if not need_tools:
                break
            try:
                tool_args = loads_json(res_text)
                tool_ret = await use_tool(ctx, session, tool_args["tool"], tool_args["data"])
                tools_additional_info += f"[工具{tool_args['tool']}返回结果: {tool_ret.strip()}]\n"
            except Exception as exc:
                logger.info("工具调用失败: %s", exc)
                break

    except openai.APIError as e:
        if session:
            logger.print_exc(f"会话 {session.id} 失败")
            if session_id_backup:
                sessions[str(session_id_backup)] = session
        ret = truncate(f"会话失败: {e.message}", 128)
        return await ctx.asend_reply_msg(ret)

    except Exception as error:
        if session:
            logger.print_exc(f"会话 {session.id} 失败")
            if session_id_backup:
                sessions[str(session_id_backup)] = session
            ret = truncate(f"会话失败: {error}", 128)
            return await ctx.asend_reply_msg(ret)
        return

    reasoning_text = ""
    if reasoning and reasoning.strip():
        if config.get("output_reasoning_content"):
            reasoning_text = f"【思考】\n{reasoning}\n【回答】\n"
        else:
            reasoning_text = f"(已思考{len(reasoning)}字)\n"

    additional_info = f"{resp_model.get_full_name()} | {total_seconds:.1f}s, {total_ptokens}+{total_ctokens} tokens"
    if rest_quota > 0:
        price_unit = resp_model.get_price_unit()
        if total_cost == 0.0:
            additional_info += f" | 0/{rest_quota:.2f}{price_unit}"
        elif total_cost >= 0.0001:
            additional_info += f" | {total_cost:.4f}/{rest_quota:.2f}{price_unit}"
        else:
            additional_info += f" | <0.0001/{rest_quota:.2f}{price_unit}"
    additional_info = f"\n({additional_info})"
    final_text = tools_additional_info + reasoning_text + res_text + additional_info

    ret = await ctx.asend_fold_msg_adaptive(final_text)
    if ret:
        ret_id = str(ret["message_id"])
        sessions[ret_id] = session
        logger.info("会话%s加入会话历史:%s, 长度:%s", session.id, ret_id, len(session))
        session.limit_length(SESSION_LEN_LIMIT_CFG.get())

    for k, v in list(sessions.items()):
        if datetime.now() - v.update_time > SESSION_EXPIRE_TIME:
            sessions.pop(k)
            logger.info("删除过期的会话%s", k)


change_model = CmdHandler(
    [
        "/模型",
        "/聊天模型",
        "/chat_model",
        "/chat model",
        "/chatmodel",
    ],
    logger,
)
change_model.check_cdrate(chat_cd).check_wblist(gwl)


@change_model.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()
    if not args:
        text_model_name = get_model_name(ctx.event, "text")
        mm_model_name = get_model_name(ctx.event, "mm")
        tool_model_name = get_model_name(ctx.event, "tool")
        image_model_name = get_model_name(ctx.event, "image")
        return await ctx.asend_reply_msg(
            f"文本模型: {text_model_name}\n"
            f"多模态模型: {mm_model_name}\n"
            f"工具模型: {tool_model_name}\n"
            f"图片生成模型: {image_model_name}"
        )

    if is_group_msg(ctx.event) and not check_superuser(ctx.event):
        return

    if "text" in args:
        last_model_name = get_model_name(ctx.event, "text")
        args = args.replace("text", "").strip()
        name = change_model_name(ctx.event, args, "text")
        return await ctx.asend_reply_msg(f"已切换文本模型: {last_model_name} -> {name}")
    if "mm" in args:
        last_model_name = get_model_name(ctx.event, "mm")
        args = args.replace("mm", "").strip()
        name = change_model_name(ctx.event, args, "mm")
        return await ctx.asend_reply_msg(f"已切换多模态模型: {last_model_name} -> {name}")
    if "tool" in args:
        last_model_name = get_model_name(ctx.event, "tool")
        args = args.replace("tool", "").strip()
        name = change_model_name(ctx.event, args, "tool")
        return await ctx.asend_reply_msg(f"已切换工具模型: {last_model_name} -> {name}")
    if "image" in args:
        last_model_name = get_model_name(ctx.event, "image")
        args = args.replace("image", "").strip()
        name = change_model_name(ctx.event, args, "image")
        return await ctx.asend_reply_msg(f"已切换图片生成模型: {last_model_name} -> {name}")

    msg = ""
    try:
        last_mm_model_name = get_model_name(ctx.event, "mm")
        name = change_model_name(ctx.event, args, "mm")
        msg += f"已切换多模态模型: {last_mm_model_name} -> {name}\n"
    except Exception as e:
        msg += f"{e}, 仅切换文本模型\n"
    last_text_model_name = get_model_name(ctx.event, "text")
    name = change_model_name(ctx.event, args, "text")
    msg += f"已切换文本模型: {last_text_model_name} -> {name}"
    return await ctx.asend_reply_msg(msg.strip())


clear_model = CmdHandler(
    [
        "/重置模型",
        "/清空模型",
        "/clear model",
        "/reset model",
        "/model reset",
        "/model clear",
    ],
    logger,
)
clear_model.check_cdrate(chat_cd).check_wblist(gwl)


@clear_model.handle()
async def _(ctx: HandlerContext):
    if is_group_msg(ctx.event) and not check_superuser(ctx.event):
        return
    clear_model_name(ctx.event)
    return await ctx.asend_reply_msg("已清空模型设置")


all_model = CmdHandler(
    [
        "/模型列表",
        "/model_list",
        "/model list",
        "/modellist",
        "/allmodel",
        "/all model",
        "/all_model",
    ],
    logger,
)
all_model.check_cdrate(chat_cd).check_wblist(gwl)


@all_model.handle()
async def _(ctx: HandlerContext):
    msg = "可用模型列表:\n"
    for model in api_provider_mgr.get_all_models():
        msg += f"{model.get_full_name()} "
        if model.input_pricing + model.output_pricing < 1e-9:
            msg += "🆓"
        if model.is_multimodal:
            msg += "🏞️"
        if model.image_response:
            msg += "🎨"
        msg += "\n"
    return await ctx.asend_fold_msg_adaptive(msg.strip())


chat_providers = CmdHandler(
    ["/供应商", "/chat_provider", "/chat provider", "/chatprovider"],
    logger,
)
chat_providers.check_cdrate(chat_cd).check_wblist(gwl)


@chat_providers.handle()
async def _(ctx: HandlerContext):
    providers = api_provider_mgr.get_all_providers()
    msg = ""
    for provider in providers:
        quota = await provider.aget_current_quota()
        msg += f"{provider.name}({provider.code}) {quota:.4f}{provider.get_price_unit()}\n"
    return await ctx.asend_reply_msg(msg.strip())


tts_request = CmdHandler(["/tts"], logger)
tts_request.check_cdrate(tts_cd).check_wblist(gwl)


@tts_request.handle()
async def _(ctx: HandlerContext):
    text = ctx.get_args().strip()
    if not text:
        return
    with TempFilePath("mp3", remove_after=timedelta(minutes=3)) as path:
        await tts(text, path)
        return await ctx.asend_msg(f"[CQ:record,file=file://{path}]")


trans = CmdHandler(["/trans", "/translate", "/翻译"], logger)
trans.check_cdrate(img_trans_cd).check_wblist(gwl)


@trans.handle()
async def _(ctx: HandlerContext):
    reply_msg = await ctx.aget_reply_msg()
    if not reply_msg:
        text = ctx.get_args().strip()
        assert_and_reply(text, "请输入要翻译的文本，或回复要翻译的文本/图片")
        return await ctx.asend_fold_msg_adaptive(await translate_text(text, cache=False))

    cqs = extract_cq_code(reply_msg)
    imgs = cqs.get("image", [])
    if not imgs:
        text = extract_text(reply_msg)
        assert_and_reply(text, "请输入要翻译的文本，或回复要翻译的文本/图片")
        return await ctx.asend_fold_msg_adaptive(await translate_text(text, cache=False))

    raise ReplyException("图片翻译器已废弃，请直接使用聊天功能翻译图片")


autochat_usermemory = CmdHandler(
    ["/autochat um", "/um", "/autochat usermemory", "/usermemory"],
    logger,
)
autochat_usermemory.check_cdrate(chat_cd).check_wblist(autochat_gwl)


@autochat_usermemory.handle()
async def _(ctx: HandlerContext):
    qids = ctx.get_at_qids()
    qid = qids[0] if qids else ctx.user_id
    nickname = await get_group_member_name(ctx.group_id, qid, bot=ctx.bot)

    um = None
    path = Path(f"data/chat/autochat/memory_{ctx.group_id}.json")
    if path.exists():
        mem = loads_json(path.read_text(encoding="utf-8"))
        um = mem.get("ums", {}).get(str(qid), {})

    if not um:
        return await ctx.asend_reply_msg(f"对@{nickname}的记忆: 无")

    um_text = f"对@{nickname}的记忆\n"
    if names := um.get("names"):
        um_text += f"🏷️ 【曾用名】\n{', '.join(names)}\n"
    if profile := um.get("profile"):
        um_text += f"👤 【用户画像】\n{profile}\n"
    if recent_events := um.get("recent_events"):
        um_text += "📅 【近期事件】\n"
        for t, event in recent_events:
            formated_time = datetime.fromtimestamp(t).strftime("%m-%d %H:%M")
            um_text += f"[{formated_time}] {event}\n"
    return await ctx.asend_fold_msg_adaptive(um_text.strip())
