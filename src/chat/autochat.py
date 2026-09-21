from datetime import datetime, timedelta
import time

from src.core import NoticeEvent, on_notice
from src.llm import ChatSession, ChatSessionResponse, get_text_embedding, tts
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


def _normalize_for_autochat(segs: list[dict]) -> list[dict]:
    """把喂给 autochat 微服务的消息段规整成它期望的形态：
    - image 段的 sub_type 转成 int（原版 serve.py 用 `== 0` 判定，字符串 "0" 会判错）
    - image 段缺 file_unique 时补一个稳定的唯一 id：优先取 url 里的 fileid 参数
      （= QQ 图片真实 file_unique，格式如 Eh...CAQJneg，和 Luna 一致），
      否则退化为 url 文件名"""
    for seg in segs:
        if seg.get("type") != "image":
            continue
        data = seg.setdefault("data", {})
        raw_sub = data.get("sub_type", data.get("subType", 0))
        try:
            raw_sub = int(raw_sub or 0)
        except (TypeError, ValueError):
            raw_sub = 0
        data["sub_type"] = raw_sub
        if not data.get("file_unique"):
            data["file_unique"] = _pick_image_id(data)
    return segs


def _pick_image_id(data: dict) -> str:
    """从 image 段里挑一个稳定的唯一 id（优先 url 的 fileid，退化为 url 文件名）。"""
    url = data.get("url")
    if not url:
        return ""
    # QQ 图片真实 id 藏在 url 的 fileid 参数里，形如 Eh...CAQJneg
    try:
        key = "fileid="
        i = url.find(key)
        if i != -1:
            fu = url[i + len(key):].split("&")[0].strip()
            if fu:
                return fu
    except Exception:
        pass
    # 退化：取 url 最后一个路径段去掉 query 和扩展名
    return url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0].split(".")[0]


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
        "msg": _normalize_for_autochat(get_msg(event)),
    }
    for cid in message_pool:
        message_pool[cid].append(msg)


def _enqueue_autochat_item(item: dict):
    for cid in message_pool:
        message_pool[cid].append(item)


_recent_poke_keys: list[tuple[int, int, int, int]] = []


def _poke_is_dup(group_id: int, from_id: int, target_id: int, ts: float) -> bool:
    global _recent_poke_keys
    ts_i = int(ts)
    gid, fid, tid = int(group_id), int(from_id), int(target_id)
    _recent_poke_keys = [k for k in _recent_poke_keys if ts_i - k[0] <= 3]
    for t, g, f, tgt in _recent_poke_keys:
        if g == gid and f == fid and tgt == tid and abs(t - ts_i) <= 2:
            return True
    _recent_poke_keys.append((ts_i, gid, fid, tid))
    return False


async def _push_poke_event(
    bot: Bot,
    group_id: int,
    from_id: int,
    target_id: int,
    ts: float,
    from_name: str | None = None,
    target_name: str | None = None,
):
    if not chat_gwl.check_id(group_id) or not autochat_gwl.check_id(group_id):
        return
    if _poke_is_dup(group_id, from_id, target_id, ts):
        return
    if not from_name:
        from_name = await get_group_member_name(group_id, from_id, bot=bot)
    if not target_name:
        target_name = await get_group_member_name(group_id, target_id, bot=bot)
    _enqueue_autochat_item({
        "msg_id": 0,
        "time": int(ts),
        "user_id": int(from_id),
        "group_id": int(group_id),
        "nickname": from_name or str(from_id),
        "msg": [{
            "type": "poke",
            "data": {
                "target_id": int(target_id),
                "target_name": target_name or str(target_id),
            },
        }],
    })


@on_notice()
async def record_group_poke(bot: Bot, event: NoticeEvent):
    if event.notice_type != "notify" or event.sub_type != "poke":
        return
    if not event.group_id:
        return
    await _push_poke_event(
        bot,
        int(event.group_id),
        int(event.user_id),
        int(event.target_id),
        float(event.time or time.time()),
    )


RPC_SERVICE = "autochat"


def on_connect(session: RpcSession):
    """RPC 客户端连上时给它建一个消息池。同时只允许一个 autochat 微服务。"""
    if message_pool:
        others = ", ".join(message_pool)
        logger.warning("已有 autochat 客户端在线 (%s)，拒绝 %s", others, session.id)
        async_task("拒绝多余autochat连接", logger)(session.close)()
        return
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


def _autochat_tts_model_name() -> str:
    return str(config.get("chat.voice.tts_model") or "").strip()


@rpc_method(RPC_SERVICE, "synth_tts")
async def handle_synth_tts(cid: str, text: str):
    text = str(text or "").strip()
    if not text:
        raise Exception("tts 文本为空")
    model_name = _autochat_tts_model_name()
    if not model_name:
        raise Exception("未配置 autochat.yaml 的 chat.voice.tts_model")
    logger.info("自动聊天RPC客户端 %s 合成语音: %s", cid, truncate(text, 64))
    with TempFilePath("mp3", remove_after=timedelta(minutes=3)) as path:
        await tts(text, path, model_name=model_name)
        return {"path": os.path.abspath(path)}


@rpc_method(RPC_SERVICE, "poke_group_member")
async def handle_poke_group_member(cid: str, group_id: int, user_id: int):
    if not chat_gwl.check_id(group_id) or not autochat_gwl.check_id(group_id):
        logger.warning("自动聊天取消戳一戳到未启用群组 %s", group_id)
        return
    bot = get_bot()
    logger.info("自动聊天RPC客户端 %s 戳群 %s 用户 %s", cid, group_id, user_id)
    ret = await bot.poke_group_member(int(group_id), int(user_id))
    await _push_poke_event(
        bot,
        int(group_id),
        int(bot.self_id),
        int(user_id),
        time.time(),
    )
    return ret


@rpc_method(RPC_SERVICE, "set_msg_emoji_like")
async def handle_set_msg_emoji_like(cid: str, group_id: int, message_id: int, emoji_id: str):
    if not chat_gwl.check_id(group_id) or not autochat_gwl.check_id(group_id):
        logger.warning("自动聊天取消贴表情到未启用群组 %s", group_id)
        return
    bot = get_bot()
    logger.info("自动聊天RPC客户端 %s 给群 %s 消息 %s 贴表情 %s", cid, group_id, message_id, emoji_id)
    return await bot.set_msg_emoji_like(int(message_id), str(emoji_id))


@rpc_method(RPC_SERVICE, "get_group_history_msg")
async def handle_get_group_msg(cid: str, group_id: int, limit: int):
    msgs = await query_recent_msg(group_id, limit)
    ret = []
    for msg in msgs:
        if check_is_bot_reply_msg(msg["msg_id"]):
            continue
        if isinstance(msg["time"], datetime):
            msg["time"] = int(msg["time"].timestamp())
        msg["msg"] = _normalize_for_autochat(msg["msg"])
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
