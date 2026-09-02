from datetime import datetime

from src.llm import ChatSession, ChatSessionResponse, get_text_embedding
from src.record import before_record_hook
from src.record.sql import query_recent_msg
from src.utils import *
from src.utils.rpc import *

config = Config("chat.autochat")
logger = get_logger("chat")
file_db = get_file_db("data/chat/db.json")

chat_gwl = get_group_white_list(file_db, logger, "chat")
autochat_gwl = get_group_white_list(file_db, logger, "autochat", is_service=False)

message_pool: dict[str, list[dict]] = {}


# ======================= 逻辑处理 ======================= #


@before_record_hook
async def record_new_message(bot: Bot, event: MessageEvent):
    """自动聊天开启的群，把新消息塞进 RPC 客户端的消息池。"""
    if not is_group_msg(event):
        return
    if not chat_gwl.check_id(event.group_id):
        return
    if not autochat_gwl.check_id(event.group_id):
        return
    msg = {
        "msg_id": event.message_id,
        "time": event.time,
        "user_id": event.user_id,
        "group_id": event.group_id,
        "nickname": get_user_name_by_event(event),
        "msg": get_msg(event),
    }
    for cid in message_pool:
        message_pool[cid].append(msg)


RPC_SERVICE = "autochat"


def on_connect(session: RpcSession):
    """RPC 客户端连上时给它建一个消息池。"""
    message_pool[session.id] = []


def on_disconnect(session: RpcSession):
    """RPC 客户端断开时清掉它的消息池。"""
    message_pool.pop(session.id, None)


start_rpc_service(
    host=config.get("rpc.host"),
    port=config.get("rpc.port"),
    token=config.get("rpc.token"),
    name=RPC_SERVICE,
    logger=logger,
    on_connect=on_connect,
    on_disconnect=on_disconnect,
)


async def _get_all_bot_group_list() -> list[dict]:
    """汇总所有 bot 的群列表，按 group_id 去重。"""
    groups = []
    seen: set[int] = set()
    for bot in iter_bots():
        try:
            for g in await bot.get_group_list() or []:
                gid = int(g.get("group_id"))
                if gid in seen:
                    continue
                seen.add(gid)
                groups.append(g)
        except Exception:
            logger.print_exc(f"获取 bot {bot.self_id} 群列表失败")
    return groups


@rpc_method(RPC_SERVICE, "get_self_info")
async def handle_get_self_info(cid: str, group_id: int):
    bot = get_bot()
    return {
        "self_id": int(bot.self_id),
        "nickname": await get_group_member_name(group_id, int(bot.self_id), bot=bot),
    }


@rpc_method(RPC_SERVICE, "get_group_list")
async def handle_get_group_list(cid: str):
    group_ids = set(chat_gwl.get()).intersection(autochat_gwl.get())
    return [g for g in await _get_all_bot_group_list() if int(g["group_id"]) in group_ids]


@rpc_method(RPC_SERVICE, "send_group_msg")
async def handle_send_group_msg(cid: str, group_id: int, message: list[dict] | str):
    if not chat_gwl.check_id(group_id) or not autochat_gwl.check_id(group_id):
        logger.warning("自动聊天取消发送消息到未启用群组 %s", group_id)
        return
    bot = get_bot()
    logger.info("自动聊天RPC客户端 %s 发送消息到群 %s: %s", cid, group_id, message)
    return await bot.send_group_msg(group_id=int(group_id), message=message)


@rpc_method(RPC_SERVICE, "get_group_history_msg")
async def handle_get_group_msg(cid: str, group_id: int, limit: int):
    msgs = await query_recent_msg(group_id, limit)
    ret = []
    for msg in msgs:
        if check_is_bot_reply_msg(msg["msg_id"]):
            continue
        if isinstance(msg["time"], datetime):
            msg["time"] = int(msg["time"].timestamp())
        ret.append(msg)
    return ret


@rpc_method(RPC_SERVICE, "query_llm")
async def handle_query_llm(cid: str, model: str | list[str], text: str, images: list[str], options: dict):
    timeout: int = options.get("timeout", 300)
    max_tokens: int = options.get("max_tokens", 2048)
    json_reply: bool = options.get("json_reply", False)
    json_key_restraints: list[dict] = options.get("json_key_restraints", [])

    imgs = []
    for img in images:
        if isinstance(img, str) and img.startswith("http"):
            img = await download_image_to_b64(img)
        imgs.append(img)

    session = ChatSession()
    session.append_user_content(text, imgs, verbose=False)

    def process(resp: ChatSessionResponse) -> str | dict:
        text = resp.result
        if not json_reply:
            return text
        try:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            text = text[start_idx:end_idx + 1]
            data = loads_json(text)
        except Exception:
            raise Exception("解析回复为json失败")
        for restraint in json_key_restraints:
            key = restraint["key"]
            dtypes = restraint.get("type")
            if isinstance(dtypes, str):
                dtypes = [dtypes]
            min_length = restraint.get("min_length")
            max_length = restraint.get("max_length")
            key = key.split(".")
            value = data
            for k in key:
                if k not in value:
                    raise Exception(f"回复的json缺少字段: {restraint['key']}")
                value = value[k]
            if dtypes and not any(isinstance(value, eval(dt)) for dt in dtypes):
                raise Exception(f"字段 {restraint['key']} 类型错误，期望类型: {dtypes}")
            if isinstance(value, (str, list)):
                if min_length and len(value) < min_length:
                    raise Exception(f"字段 {restraint['key']} 长度过短，最小长度: {min_length}")
                if max_length and len(value) > max_length:
                    raise Exception(f"字段 {restraint['key']} 长度过长，最大长度: {max_length}")
        return data

    logger.info("自动聊天RPC客户端 %s 请求LLM模型", cid)
    return await session.get_response(
        model_name=model,
        process_func=process,
        timeout=timeout,
        max_tokens=max_tokens,
    )


@rpc_method(RPC_SERVICE, "query_embedding")
async def handle_query_embedding(cid: str, texts: list[str], model_name: str):
    logger.info("自动聊天RPC客户端 %s 请求 %s 条文本嵌入", cid, len(texts))
    return await get_text_embedding(texts, model_name)


@rpc_method(RPC_SERVICE, "get_new_msgs")
async def handle_get_new_msgs(cid: str):
    if cid not in message_pool:
        return []
    msgs = message_pool.get(cid, [])
    message_pool[cid] = []
    return msgs
