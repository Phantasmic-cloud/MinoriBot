try:
    from .utils import *
    from .memory import *
except ImportError:
    from utils import *
    from memory import *
import re
import json
import numpy as np


def debug_mode() -> bool:
    return config.get('log_level') == 'DEBUG'


# ================ RPC接口定义 ================= #

@dataclass
class Message:
    msg_id: int
    time: datetime
    user_id: int
    group_id: int
    nickname: str
    msg: list[dict]

rpc_session = RpcSession(
    config.item('rpc.host'), 
    config.item('rpc.port'),
    config.item('rpc.token'),
    config.item('rpc.reconnect_interval'),
)

async def rpc_get_self_info(group_id: int):
    return await rpc_session.call('get_self_info', group_id)

async def rpc_send_group_msg(group_id: int, message: str):
    return await rpc_session.call('send_group_msg', group_id, message)

async def rpc_query_llm(model: str, prompt: str, images: list[dict] = [], options: dict = {}):
    return await rpc_session.call('query_llm', model, prompt, images, options, timeout=options.get('timeout', 300) + 5)

async def rpc_get_group_history_msg(group_id: int, limit: int) -> list[Message]:
    msgs = await rpc_session.call('get_group_history_msg', group_id, limit)
    return [Message(
        msg_id=msg['msg_id'],
        time=datetime.fromtimestamp(msg['time']),
        user_id=msg['user_id'],
        group_id=group_id,
        nickname=msg['nickname'],
        msg=msg['msg'],
    ) for msg in msgs]

async def rpc_get_new_msgs():
    msgs = await rpc_session.call('get_new_msgs')
    return [Message(
        msg_id=msg['msg_id'],
        time=datetime.fromtimestamp(msg['time']),
        user_id=msg['user_id'],
        group_id=msg['group_id'],
        nickname=msg['nickname'],
        msg=msg['msg'],
    ) for msg in msgs]

async def rpc_query_embeddings(texts: list[str], model_name: str) -> list[list[float]]:
    return await rpc_session.call('query_embedding', texts, model_name, timeout=60)





# ================ 处理逻辑 ================= #

file_db = get_file_db("data/chat/autochat/db.json")

# ================ Sticker搜索 ================= #

# caption向量缓存: list of (sid, text, path, full_emb, emotion_emb)
_sticker_cache: list[tuple[int, str, str, np.ndarray, np.ndarray]] = []
_sticker_cache_mtime: float = 0.0

STK_EMB_DB_PATH = "data/chat/autochat/stk_emb_db.json"

def _load_emb_db(model: str) -> dict:
    """读取持久化向量库，模型不一致则清空返回空dict"""
    try:
        with open(STK_EMB_DB_PATH, 'r', encoding='utf-8') as f:
            db = json.load(f)
        if db.get('emb_model') != model:
            info(f"Embedding模型已变更({db.get('emb_model')} -> {model})，清空向量库")
            return {}
        return db.get('embeddings', {})
    except FileNotFoundError:
        return {}
    except Exception as e:
        warning(f"读取stk_emb_db.json失败，重新建立: {get_exc_desc(e)}")
        return {}

def _save_emb_db(model: str, embeddings: dict):
    """保存向量库到文件"""
    os.makedirs(os.path.dirname(STK_EMB_DB_PATH), exist_ok=True)
    with open(STK_EMB_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump({'emb_model': model, 'embeddings': embeddings}, f, ensure_ascii=False)

# 按群隔离的表情包倍率: {group_id: {sid: float}}
sticker_multipliers: dict[int, dict[int, float]] = {}

def get_sticker_multiplier(group_id: int, sid: int) -> float:
    return sticker_multipliers.get(group_id, {}).get(sid, 1.0)

def update_sticker_multipliers(group_id: int, sent_sid: int | None, all_sids: list[int]):
    send_penalty = config.get('chat.sticker.send_penalty')
    recover_rate = config.get('chat.sticker.recover_rate')
    if group_id not in sticker_multipliers:
        sticker_multipliers[group_id] = {}
    m = sticker_multipliers[group_id]
    for sid in all_sids:
        if sid == sent_sid:
            m[sid] = max(0.0, m.get(sid, 1.0) - send_penalty)
        else:
            m[sid] = min(1.0, m.get(sid, 1.0) + recover_rate)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

async def _build_sticker_cache(model: str):
    global _sticker_cache, _sticker_cache_mtime
    db_path = "data/chat/autochat/sticker_db.json"
    mtime = os.path.getmtime(db_path)
    with open(db_path, 'r', encoding='utf-8') as f:
        db = json.load(f)
    stickers = db.get('stickers', {})
    sid_texts: list[tuple[int, str, str]] = []
    for sid_str, s in stickers.items():
        sid = int(sid_str)
        path = os.path.abspath(s.get('path', ''))
        captions = s.get('caption', [])
        if isinstance(captions, str):
            captions = [{'emotion': '', 'scene': captions}]
        for c in captions:
            emotion = c.get('emotion', '')
            scene = c.get('scene', '')
            if emotion or scene:
                text = f"{emotion},{scene}" if emotion else scene
                sid_texts.append((sid, text, path))
    if not sid_texts:
        _sticker_cache = []
        _sticker_cache_mtime = mtime
        return
    # 从持久化向量库读取已有向量
    emb_db = _load_emb_db(model)
    new_emb_db = {}  # 最终要写回的向量库

    # 找出哪些需要重新请求
    caption_texts = [t for _, t, _ in sid_texts]
    emotion_texts = [t.split(',')[0] for t in caption_texts]
    keys = [f"{sid}:{text}" for sid, text, _ in sid_texts]

    missing_indices = [i for i, k in enumerate(keys) if k not in emb_db or f"e:{k}" not in emb_db]
    if missing_indices:
        success_count = 0
        # 分批请求，每批单独try/catch，失败跳过不影响其他批
        for batch_start in range(0, len(missing_indices), 10):
            batch_idx = missing_indices[batch_start:batch_start+10]
            try:
                batch_full_texts = [caption_texts[i] for i in batch_idx]
                batch_emotion_texts = [emotion_texts[i] for i in batch_idx]
                full_embs = await rpc_query_embeddings(batch_full_texts, model)
                emotion_embs_batch = await rpc_query_embeddings(batch_emotion_texts, model)
                for i, full_e, emotion_e in zip(batch_idx, full_embs, emotion_embs_batch):
                    emb_db[keys[i]] = full_e
                    emb_db[f"e:{keys[i]}"] = emotion_e
                success_count += len(batch_idx)
            except Exception as e:
                warning(f"Sticker向量批次请求失败，跳过{len(batch_idx)}条，下次重试: {get_exc_desc(e)}")
        if success_count > 0:
            info(f"新增{success_count}/{len(missing_indices)}条向量")

    # 构建缓存，同时收集本次用到的key
    caption_embs = [emb_db[k] for k in keys]
    emotion_embs = [emb_db[f"e:{k}"] for k in keys]
    _sticker_cache = [
        (sid, text, path, np.array(emb, dtype=np.float32), np.array(eemb, dtype=np.float32))
        for (sid, text, path), emb, eemb in zip(sid_texts, caption_embs, emotion_embs)
    ]
    # 只保留本次用到的key，清理已删除表情包的向量
    used_keys = set(keys) | {f"e:{k}" for k in keys}
    new_emb_db = {k: v for k, v in emb_db.items() if k in used_keys}
    _save_emb_db(model, new_emb_db)
    is_update = _sticker_cache_mtime > 0
    _sticker_cache_mtime = mtime
    info(f"Sticker{'更新' if is_update else '缓存'}完成，共{len(_sticker_cache)}条向量")

async def search_sticker(group_id: int, query: dict) -> tuple[str, int] | tuple[None, None]:
    global _sticker_cache, _sticker_cache_mtime
    db_path = "data/chat/autochat/sticker_db.json"
    threshold = config.get('chat.sticker.similarity_threshold')
    model = config.get('chat.sticker.emb_model')
    sticker_emotion = query.get('emotion', '')
    sticker_scene = query.get('scene', '')
    query = f"{sticker_emotion},{sticker_scene}" if sticker_emotion else sticker_scene
    try:
        # 检查缓存是否需要更新
        try:
            mtime = os.path.getmtime(db_path)
            if mtime > _sticker_cache_mtime:
                await _build_sticker_cache(model)
        except FileNotFoundError:
            return None, None, None, None
        if not _sticker_cache:
            return None, None, None, None
        # query拆emotion和full，两次请求
        query_embs = await rpc_query_embeddings([query, sticker_emotion], model)
        query_full_emb = np.array(query_embs[0], dtype=np.float32)
        query_emotion_emb = np.array(query_embs[1], dtype=np.float32)
        # 第一阶段：emotion相似度过滤，低于threshold的排除
        # 第二阶段：在通过的候选里用full相似度选最高
        sid_best: dict[int, tuple[float, str]] = {}
        for sid, text, path, full_emb, emotion_emb in _sticker_cache:
            emotion_score = cosine_similarity(query_emotion_emb, emotion_emb)
            if emotion_score < threshold:
                continue
            full_score = cosine_similarity(query_full_emb, full_emb)
            if sid not in sid_best or full_score > sid_best[sid][0]:
                sid_best[sid] = (full_score, path)
        # 乘以倍率后选最高
        all_sids = list(sid_best.keys())
        old_multipliers = {sid: get_sticker_multiplier(group_id, sid) for sid in all_sids}
        best_path, best_score, best_sid = None, -1.0, None
        for sid, (raw_score, path) in sid_best.items():
            adjusted = raw_score * old_multipliers[sid]
            if adjusted > best_score:
                best_score = adjusted
                best_path = path
                best_sid = sid
        if best_score >= threshold:
            raw = sid_best[best_sid][0]
            info(f"Sticker: sid={best_sid}, adjusted={raw:.3f}*{old_multipliers[best_sid]:.1f}={best_score:.3f}, query={query}")
            return best_path, best_sid, all_sids, old_multipliers
        else:
            info(f"没有合适的表情包，取消发送")
            update_sticker_multipliers(group_id, None, all_sids)
            changed = [f"sid={s}({old_multipliers[s]:.1f}→{get_sticker_multiplier(group_id, s):.1f})"
                       for s in all_sids if abs(get_sticker_multiplier(group_id, s) - old_multipliers[s]) > 0.001]
            if changed:
                info(f"表情包倍率: {', '.join(changed)}")
            return None, None, None, None
    except BaseException as e:
        warning(f"Sticker搜索内部失败: {get_exc_desc(e)}")
        return None, None, None, None


@dataclass
class GroupStatus:
    group_id: int
    willingness: float
    self_msg_ids: list[int]
    last_check_willing_time: float
    last_reply_time: float
    
    @staticmethod
    def load(group_id):
        data = file_db.get(f'status_{group_id}', {})
        return GroupStatus(
            group_id=group_id,
            willingness=data.get('willingness', 0.0),
            self_msg_ids=data.get('self_msg_ids', []),
            last_check_willing_time=data.get('last_check_willing_time', None),
            last_reply_time=data.get('last_reply_time', None),
        )
    
    def save(self):
        file_db.set(f'status_{self.group_id}', {
            'willingness': self.willingness,
            'self_msg_ids': self.self_msg_ids,
            'last_check_willing_time': self.last_check_willing_time,
            'last_reply_time': self.last_reply_time,
        })


group_mems: dict[int, MemorySystem] = {}

def get_group_memory_system(group_id: int) -> MemorySystem:
    if group_id not in group_mems:
        group_mems[group_id] = MemorySystem("data/chat/autochat", group_id)
    return group_mems[group_id]


image_caption_db = get_file_db("data/chat/autochat/image_captions.json")

async def get_image_caption(data: dict, use_llm: bool) -> str:
    summary = data.get("summary", '')
    url = data.get("url", None)
    file_unique = data.get("file_unique", '')
    sub_type = data.get("sub_type", 0)
    sub_type = "图片" if sub_type == 0 else "表情"
    fallback_caption = f"[{sub_type}]" if not summary else f"[{sub_type}:{summary}]"

    info(f"尝试获取图片总结: file_unique={file_unique} subtype={sub_type} url={url} summary={summary}")
    if file_unique:
        cache = image_caption_db.get(file_unique)
        if cache:
            info(f"图片总结命中缓存: {cache}")
            return f"[{sub_type}:{cache}]"
        
    try:
        if not use_llm:
            return fallback_caption
        caption = await rpc_query_llm(
            model=config.get('image_caption.model'),
            prompt=config.get('image_caption.prompt').format(sub_type=sub_type),
            images=[url],
            options={ 
                'timeout': config.get('image_caption.timeout'),
                'max_tokens': config.get('image_caption.max_tokens'),
            }
        )
        assert caption, "图片总结为空"

        info(f"图片总结成功: {caption}")
        if file_unique:
            image_caption_db.set(file_unique, caption)
        return f"[{sub_type}:{caption}]"
    
    except Exception as e:
        warning(f"总结图片 url={url} 失败: {get_exc_desc(e)}")
        return fallback_caption
        
def json_msg_to_readable_text(data: dict):
    try:
        data = loads_json(data['data'])
        title = data["meta"]["detail_1"]["title"]
        desc = truncate(data["meta"]["detail_1"]["desc"], 32)
        url = data["meta"]["detail_1"]["qqdocurl"]
        return f"[{title}分享:{desc}]"
    except:
        try:
            return f"[转发消息:{data['prompt']}]"
        except:
            return "[转发消息]"

async def format_msgs(
    msgs: list[Message], 
    image_caption_limit: int, 
    image_caption_prob: float,
    emotion_caption_limit: int,
    emotion_caption_prob: float,
) -> str:
    msgs = sorted(msgs, key=lambda m: m.time, reverse=True)
    texts = []
    captioned_images = 0
    captioned_emotions = 0
    for msg in msgs:
        text = f"{get_readable_datetime(msg.time)} [{msg.msg_id}] {msg.nickname}({msg.user_id}):\n"
        for seg in msg.msg:
            stype, sdata = seg['type'], seg['data']
            match stype:
                case "text":
                    text += sdata['text']
                case "face":
                    text += "[表情]"
                case "video":
                    text += "[视频]"
                case "audio":
                    text += "[音频]"
                case "file":
                    text += "[文件]"
                case "at":
                    text += f"[@{sdata['qq']}]"
                case "reply":
                    text += f"[reply={sdata['id']}]"
                case "forward":
                    text += "[转发聊天记录]"
                case "json":
                    text += json_msg_to_readable_text(sdata)
                case "image":
                    if sdata.get("sub_type", 0) == 0:
                        text += await get_image_caption(
                            sdata,
                            captioned_images < image_caption_limit and random.random() < image_caption_prob,
                        )
                        captioned_images += 1
                    else:
                        text += await get_image_caption(
                            sdata,
                            captioned_emotions < emotion_caption_limit and random.random() < emotion_caption_prob,
                        )
                        captioned_emotions += 1
        texts.append(text.strip())
    return "\n".join(reversed(texts))

def get_plain_text(msg: Message) -> str:
    ret = ""
    for seg in msg.msg:
        if seg['type'] == 'text':
            ret += seg['data']['text']
    return ret.strip()


async def generate_summary(text: str) -> str:
    try:
        info(f"开始生成文本摘要: {truncate(text, 20)}")
        summary = await rpc_query_llm(
            model=config.get('summary.model'),
            prompt=config.get('summary.prompt').format(text=text),
            images=[],
            options={
                'timeout': config.get('summary.timeout'),
                'max_tokens': config.get('summary.max_tokens'),
            }
        )
        info(f"生成文本摘要成功: {summary}")
        return summary
    except Exception as e:
        error(f"生成摘要失败: {e}")
        return ""



# ================ 主聊天逻辑 ================= #
    
_self_infos: dict[int, dict] = {}

async def chat(msg: Message):
    if msg.group_id not in _self_infos:
        _self_infos[msg.group_id] = await rpc_get_self_info(msg.group_id)
    self_id = int(_self_infos[msg.group_id]['self_id'])
    self_name = _self_infos[msg.group_id]['nickname']

    if msg.user_id == self_id:  # 自己发的消息不触发
        return
    if get_plain_text(msg).startswith("/"):  # 命令消息不触发
        return

    status = GroupStatus.load(msg.group_id)
    # 忽略上次回复思考时接收到的消息
    if status.last_reply_time and msg.time.timestamp() <= status.last_reply_time:
        return
    
    info(f"{msg.group_id} 的新消息 {msg.msg_id} {msg.nickname}({msg.user_id}): {get_plain_text(msg)}")
    
    # ---------------- 更新意愿值 ---------------- #

    try:
        delta = 0.0
        # 随时间减少
        if status.last_check_willing_time:
            time_passed = time.time() - status.last_check_willing_time
            delta -= min(config.get('chat.willing.decrease_per_minute') * time_passed / 60.0, status.willingness)
        # 每条消息增加
        delta += config.get('chat.willing.increase_per_msg')
        # 基于消息内容调整（@ 和回复可以同时生效，各自最多加一次）
        got_at = False
        got_reply = False
        for seg in msg.msg:
            stype, sdata = seg['type'], seg['data']
            if not got_at and stype == 'at' and int(sdata['qq']) == self_id:
                delta += config.get('chat.willing.increase_per_at')
                got_at = True
            if not got_reply and stype == 'reply' and int(sdata['id']) in status.self_msg_ids:
                delta += config.get('chat.willing.increase_per_reply')
                got_reply = True
            if got_at and got_reply:
                break
        # 基于关键字调整
        plain_text = get_plain_text(msg).lower()
        for kw, value in config.get('chat.willing.increase_keywords').items():
            if kw.lower() in plain_text:
                delta += value
        # 群组调整
        delta *= config.get('chat.willing.group_scale').get(str(msg.group_id), 1.0)
        last_willingness = status.willingness
        status.willingness += delta
        status.willingness = min(status.willingness, config.get('chat.willing.limit'))
        status.last_check_willing_time = time.time()
        status.save()

        reply_rate = min(max(status.willingness, 0.0), 1.0)
        if random.random() > reply_rate:
            info(f"意愿值: {last_willingness:.4f} -> {status.willingness:.4f}")
            return
        
        info(f"意愿值: {last_willingness:.4f} -> {status.willingness:.4f}, 决定回复该消息")
        await asyncio.sleep(config.get('chat.get_history_msg_delay_seconds'))

    except:
        error(f"更新意愿值时失败，放弃聊天处理")
        return
    
    info("=" * 20)
    info(f"开始对消息 {msg.msg_id} 进行聊天处理")

    # ---------------- 消息处理 ---------------- #
    try:
        recent_msgs = await rpc_get_group_history_msg(msg.group_id, config.get('chat.history_msg_num'))
        # 隐藏所有命令消息
        recent_msgs = [m for m in recent_msgs if not get_plain_text(m).startswith("/")]
        # 如果历史消息中没有当前消息，则添加
        if not any(m.msg_id == msg.msg_id for m in recent_msgs):
            recent_msgs.append(msg)
        info(f"获取最近共 {len(recent_msgs)} 条有效聊天记录")

        recent_text = await format_msgs(
            recent_msgs,
            image_caption_limit=config.get('image_caption.image_limit'),
            image_caption_prob=config.get('image_caption.image_prob'),
            emotion_caption_limit=config.get('image_caption.emotion_limit'),
            emotion_caption_prob=config.get('image_caption.emotion_prob'),
        )
        recent_summary = await generate_summary(recent_text)
        if recent_summary == "":
            warning("生成聊天记录摘要失败，放弃聊天处理")
            return

        last_long_msg = None
        for m in reversed(recent_msgs):
            msg_text = get_plain_text(m)
            if len(msg_text) >= 4:
                last_long_msg = msg_text
                break

        texts_to_embed = [recent_summary]
        if last_long_msg:
            texts_to_embed.append(last_long_msg)
        query_embs = await rpc_query_embeddings(texts_to_embed, config.get('chat.llm.emb_model'))
        recent_emb = query_embs[0]
            
    except:
        error(f"处理消息时失败，放弃聊天处理")
        return

    # ---------------- 获取记忆 ---------------- #

    try:
        mem = get_group_memory_system(msg.group_id)

        # 获取事件记忆
        short_em_num, long_em_num = config.get('chat.mem.short_em_num'), config.get('chat.mem.long_em_num')
        em_text = ""
        short_ems, long_ems = [], []
        if short_em_num + long_em_num > 0:
            short_ems = mem.em_query(query_embs, short_em_num, 'short_term', config.get('chat.mem.em_time_decay_per_hour'))
            long_ems = mem.em_query(query_embs, long_em_num, 'long_term')
            info(f"获取短期事件记忆共 {len(short_ems)} 条: {[e.id for e in short_ems]}")
            info(f"获取长期事件记忆共 {len(long_ems)} 条: {[e.id for e in long_ems]}")
            if short_ems or long_ems:
                em_text += "可能与你当前聊天内容相关的记忆事件:\n"
                em_text += "```\n"
                for em in short_ems + long_ems:
                    em_text += f"{get_readable_datetime(datetime.fromtimestamp(em.created_at))}: {em.text}\n"
                em_text += "```\n"

        # 获取自身回复记忆
        sm_num = config.get('chat.mem.sm_num')
        sm_text = ""
        if sm_num > 0:
            sms: list[SelfMemory] = mem.sm_get()[-sm_num:] if sm_num > 0 else []
            info(f"获取自身记忆共 {len(sms)} 条: {[s.id for s in sms]}")
            if sms:
                sm_text += "你自己过去的回复记录供参考:\n"
                sm_text += "```\n"
                for sm in sms:
                    sm_text += f"{get_readable_datetime(sm.time)} [{sm.id}]: {sm.text}\n"
                sm_text += "```\n"

        # 获取用户记忆
        um_num = config.get('chat.mem.um_num')
        um_text = ""
        top_user_ids = []
        if um_num > 0:
            user_msg_counts = {}
            for m in recent_msgs:
                user_msg_counts[m.user_id] = user_msg_counts.get(m.user_id, 0) + 1
            top_users = sorted(user_msg_counts.items(), key=lambda x: x[1], reverse=True)
            candidate_uids = [uid for uid, _ in top_users]
            # 包含消息中提到名称的用户（更优先）
            full_msg = "".join(get_plain_text(m) for m in recent_msgs)
            mentioned_uids = mem.um_query_uid_by_name_in_message(full_msg)
            for uid in mentioned_uids:
                if uid not in candidate_uids:
                    candidate_uids.insert(0, uid)
            # 限制数量
            candidate_uids = candidate_uids[:um_num]
            # 格式化 UserMemory
            ums_content = []
            for user_id in candidate_uids:
                top_user_ids.append(user_id)
                if um := mem.um_get(user_id):
                    u_info = f"用户ID: {user_id}\n"
                    if um.names: u_info += f"  - 曾用名: {', '.join(um.names)}\n"
                    if um.profile: u_info += f"  - 简介: {um.profile}\n"
                    if um.recent_events:
                        u_info += "  - 最近事件:\n"
                        for t, txt in um.recent_events:
                            u_info += f"    [{get_readable_datetime(datetime.fromtimestamp(t))}]: {txt}\n"
                    ums_content.append(u_info)
            if ums_content:
                um_text += "你对聊天中的部分用户的记忆:\n"
                um_text += "```\n" + "\n".join(ums_content) + "```\n"

    except:
        error(f"获取记忆时失败，放弃聊天处理")
        return

    # ---------------- 请求LLM生成回复 ---------------- #

    try:
        recent_text = f"""
以下是最近的聊天记录:
```
{recent_text}
```
""".strip()
        
        persona = config.get('chat.prompt.persona')
        if msg.group_id in persona:
            persona = persona[msg.group_id]
        else:
            persona = persona.get('default', '')
        
        full_prompt: str = config.get('chat.prompt.framework').format(
            self_id=self_id,
            self_name=self_name,
            persona=persona,
            recent_text=recent_text,
            em_text=em_text,
            sm_text=sm_text,
            um_text=um_text,
        )

        if debug_mode():
            save_dir = "sandbox/autochat_prompt.txt"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            with open(save_dir, 'w', encoding='utf-8') as f:
                f.write(full_prompt)
        
        info(f"开始请求LLM生成回复，输入长度 {len(full_prompt)} 字符")
        llm_response = await rpc_query_llm(
            model=config.get('chat.llm.model'),
            prompt=full_prompt,
            images=[],
            options={
                'timeout': config.get('chat.llm.timeout'),
                'max_tokens': config.get('chat.llm.max_tokens'),
                'json_reply': True,
                'json_key_restraints': [
                    { 'key': 'reply', 'type': 'str' },
                    { 'key': 'user_updates', 'type': 'list' },
                ],
            }
        )
        info(f"LLM生成回复成功: {llm_response}")

        reply_text = llm_response['reply']
        reply_text2 = llm_response.get('reply2', '')
        sticker_query = llm_response.get('sticker', '')
        user_updates = llm_response.get('user_updates', [])

    except:
        error(f"请求LLM生成回复时失败，放弃聊天处理")
        return

    # ---------------- 搜索表情包 ---------------- #

    sticker_path, sticker_sid, sticker_all_sids, sticker_old_multipliers = None, None, None, None
    if sticker_query and isinstance(sticker_query, dict) and (sticker_query.get('emotion') or sticker_query.get('scene')):
        try:
            sticker_path, sticker_sid, sticker_all_sids, sticker_old_multipliers = await search_sticker(msg.group_id, sticker_query)
        except BaseException as e:
            warning(f"Sticker搜索失败: {get_exc_desc(e)}")

    # ---------------- 发送回复 ---------------- #

    try:
        send_msg_id_texts: list[tuple[int, str]] = []

        async def process_reply_text(index: int, text: str):
            if not text:
                info(f"LLM生成的回复{index}为空，放弃发送")
                return
            # 获取at和回复
            at_id, reply_id = None, None
            # 匹配 [@id]
            if at_match := re.search(r"\[@(\d+)\]", text):
                at_id = int(at_match.group(1))
                text = text.replace(at_match.group(0), "")
                if any(m.user_id == at_id for m in recent_msgs):
                    text = f"[CQ:at,qq={at_id}]" + text
            # 匹配 [reply=id]
            if reply_match := re.search(r"\[reply=(\d+)\]", text):
                reply_id = int(reply_match.group(1))
                text = text.replace(reply_match.group(0), "")
                if any(m.msg_id == reply_id for m in recent_msgs):
                    text = f"[CQ:reply,id={reply_id}]" + text
            text = truncate(text, config.get('chat.reply_max_length'))
            info(f"自动聊天生成回复{index}: {text} at_id={at_id} reply_id={reply_id}")
        
            send_ret = await rpc_send_group_msg(msg.group_id, text)
            send_msg_id = int(send_ret['message_id'])
            send_msg_id_texts.append((send_msg_id, text))
            info(f"发送回复{index}成功: send_msg_id={send_msg_id}")

            status.load(msg.group_id)
            status.self_msg_ids.append(send_msg_id)
            status.self_msg_ids = status.self_msg_ids[-100:]
            status.last_reply_time = time.time()
            status.save()

        await process_reply_text(1, reply_text)
        await asyncio.sleep(config.get('chat.reply_interval_seconds'))
        await process_reply_text(2, reply_text2)

        # 发送表情包
        if sticker_path:
            await asyncio.sleep(config.get('chat.reply_interval_seconds'))
            send_ret = await rpc_send_group_msg(msg.group_id, f"[CQ:image,file=file://{sticker_path}]")
            send_msg_id = int(send_ret['message_id'])
            status.load(msg.group_id)
            status.self_msg_ids.append(send_msg_id)
            status.self_msg_ids = status.self_msg_ids[-100:]
            status.last_reply_time = time.time()
            status.save()
            info(f"表情包发送成功: sid={sticker_sid}")
            sticker_captions = [text.replace(',', '/') for sid, text, path, full_emb, emotion_emb in _sticker_cache if sid == sticker_sid]
            sticker_desc = ' | '.join(sticker_captions) if sticker_captions else f"sid={sticker_sid}"
            send_msg_id_texts.append((send_msg_id, f"[表情包: {sticker_desc}]"))
            update_sticker_multipliers(msg.group_id, sticker_sid, sticker_all_sids)
            changed = [f"sid={s}({sticker_old_multipliers[s]:.1f}→{get_sticker_multiplier(msg.group_id, s):.1f})"
                       for s in sticker_all_sids if abs(get_sticker_multiplier(msg.group_id, s) - sticker_old_multipliers[s]) > 0.001]
            if changed:
                info(f"表情包倍率: {', '.join(changed)}")

    except:
        error(f"发送回复时失败")
        return

    # ---------------- 更新意愿值 ---------------- #

    try:
        status.load(msg.group_id)
        last_willingness = status.willingness
        status.willingness *= config.get('chat.willing.decay_after_send')
        status.willingness -= config.get('chat.willing.decrease_after_send')
        status.willingness = max(status.willingness, 0.0)
        status.save()
        info(f"聊天后意愿值: {last_willingness:.4f} -> {status.willingness:.4f}")
    except:
        error(f"聊天后更新意愿值失败")

    # ---------------- 更新记忆 ---------------- #

    try:
        # 添加事件记忆
        mem.em_add(
            text=recent_summary,
            embedding=recent_emb,
            initial_weight=0.0,
        )
        # 短期记忆添加权重
        for em in short_ems:
            mem.em_increase_weight(
                memory_id=em.id,
                weight_increase=config.get('chat.mem.short_em_reward'),
                threshold=config.get('chat.mem.em_long_term_threshold'),
            )
        # 遗忘短期记忆
        mem.em_forget(
            forget_time=(datetime.now() - timedelta(days=config.get('chat.mem.short_em_forget_days'))).timestamp(),
            forget_prob=config.get('chat.mem.short_em_forget_prob'),
        )

        # 添加用户记忆
        if isinstance(user_updates, list):
            for update in user_updates:
                try:
                    uid = int(update.get('user_id'))
                    # 安全检查：只允许更新当前上下文中存在的用户，防止LLM幻觉
                    if uid not in top_user_ids and uid != msg.user_id:
                        continue
                    new_names = []
                    if update.get('new_name'):
                        new_names.append(update.get('new_name'))
                    for msg in reversed(recent_msgs):
                        if msg.user_id == uid:
                            new_names.append(msg.nickname)
                            break
                    mem.um_update(
                        user_id=uid,
                        new_names=new_names,
                        wrong_names=update.get('wrong_names'),
                        profile_update=update.get('profile'),
                        event_update=update.get('new_event'),
                        max_events=config.get('chat.mem.um_max_events'),
                        max_names=config.get('chat.mem.um_max_names'),
                    )
                except Exception as e:
                    warning(f"解析用户记忆更新失败: {update}, err={get_exc_desc(e)}")
            
        # 添加自身记忆
        for msg_id, text in send_msg_id_texts:
            mem.sm_add(
                msg_id=msg_id,
                text=text,
                keep_count=config.get('chat.mem.sm_keep_count'),
            )

    except:
        error(f"更新记忆失败")

    info(f"完成对消息 {msg.msg_id} 的聊天处理")
    info("=" * 20)
  

# ================ 主循环 ================= #

group_queues: dict[int, asyncio.Queue] = {}


async def group_message_worker(group_id: int, queue: asyncio.Queue):
    while True:
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=3*60*60)
            try:
                await chat(msg)
            except Exception as e:
                error(f"群 {group_id} 消息处理异常: {get_exc_desc(e)}")
            finally:
                queue.task_done()
        except asyncio.TimeoutError:
            if group_id in group_queues:
                del group_queues[group_id]
            info(f"群 {group_id} 闲置超时，Worker 退出")
            break


async def main():
    asyncio.create_task(rpc_session.run(reconnect=True))
    await asyncio.sleep(1)
    info("开始监听新消息")

    while True:
        await asyncio.sleep(1)
        
        msgs = []
        try:
            msgs = await rpc_get_new_msgs()
        except Exception as e:
            warning(f"获取新消息失败: {get_exc_desc(e)}")
            continue
            
        for msg in msgs:
            group_id = msg.group_id
            if group_id not in group_queues:
                queue = asyncio.Queue()
                group_queues[group_id] = queue
                asyncio.create_task(group_message_worker(group_id, queue))
            group_queues[group_id].put_nowait(msg)


if __name__ == '__main__':
    asyncio.run(main())