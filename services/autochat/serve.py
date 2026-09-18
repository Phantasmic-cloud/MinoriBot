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

async def rpc_poke_group_member(group_id: int, user_id: int):
    return await rpc_session.call('poke_group_member', group_id, user_id)

async def rpc_set_msg_emoji_like(group_id: int, message_id: int, emoji_id: str):
    return await rpc_session.call('set_msg_emoji_like', group_id, message_id, emoji_id)


MAX_ACTIONS = 5
MAX_TEXT_ACTIONS = 3
MAX_REACT_ACTIONS = 3


def _parse_poke_ids(raw) -> list[int]:
    if raw is None or raw == '':
        return []
    if isinstance(raw, (int, float, str)):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    ids: list[int] = []
    seen: set[int] = set()
    for item in items:
        try:
            uid = int(item)
        except (TypeError, ValueError):
            continue
        if uid <= 0 or uid in seen:
            continue
        seen.add(uid)
        ids.append(uid)
    return ids


def _sticker_query_ok(query) -> bool:
    return isinstance(query, dict) and bool(query.get('emotion') or query.get('scene'))


def _emoji_to_id(raw) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        eid = str(int(raw))
        return eid if eid.isdigit() and int(eid) > 0 else None
    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit():
        return text if int(text) > 0 else None
    ch = text[0]
    cp = ord(ch)
    if cp < 128:
        return None
    return str(cp)


def _parse_react(raw) -> tuple[int, str] | None:
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    try:
        msg_id = int(raw[0])
    except (TypeError, ValueError):
        return None
    emoji_id = _emoji_to_id(raw[1])
    if emoji_id is None:
        return None
    return msg_id, emoji_id


def _expand_action_item(item) -> list[dict]:
    if isinstance(item, str):
        text = item.strip()
        return [{'kind': 'text', 'text': text}] if text else []
    if not isinstance(item, dict):
        return []
    actions = []
    for key, val in item.items():
        if key == 'text':
            text = str(val).strip() if val is not None else ''
            if text:
                actions.append({'kind': 'text', 'text': text})
        elif key == 'poke':
            ids = _parse_poke_ids(val)
            if ids:
                actions.append({'kind': 'poke', 'ids': ids})
        elif key == 'sticker' and _sticker_query_ok(val):
            actions.append({'kind': 'sticker', 'query': val})
        elif key == 'react':
            react = _parse_react(val)
            if react:
                actions.append({'kind': 'react', 'msg_id': react[0], 'emoji_id': react[1]})
    return actions


def _parse_actions(llm_response: dict) -> list[dict]:
    actions: list[dict] = []
    raw = llm_response.get('actions')
    if isinstance(raw, list):
        for item in raw:
            actions.extend(_expand_action_item(item))

    out: list[dict] = []
    text_n = 0
    react_n = 0
    for action in actions:
        if len(out) >= MAX_ACTIONS:
            break
        if action['kind'] == 'text':
            if text_n >= MAX_TEXT_ACTIONS:
                continue
            text_n += 1
        elif action['kind'] == 'react':
            if react_n >= MAX_REACT_ACTIONS:
                continue
            react_n += 1
        out.append(action)
    return out

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
_sticker_job: asyncio.Task | None = None

STK_EMB_DB_PATH = "data/chat/autochat/stk_emb_db.json"
STK_DB_PATH = "data/chat/autochat/sticker_db.json"


def _sticker_job_running() -> bool:
    return _sticker_job is not None and not _sticker_job.done()


def _read_stk_emb_file() -> dict | None:
    try:
        with open(STK_EMB_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        warning(f"读取stk_emb_db.json失败，重新建立: {get_exc_desc(e)}")
        return None


def _emb_db_usable(model: str) -> tuple[bool, dict]:
    raw = _read_stk_emb_file()
    if raw is None:
        return False, {}
    stored = raw.get('emb_model')
    if stored != model:
        info(f"Embedding模型已变更({stored} -> {model})，清空向量库")
        return False, {}
    emb = raw.get('embeddings') or {}
    return True, emb if isinstance(emb, dict) else {}


def _save_emb_db(model: str, embeddings: dict):
    os.makedirs(os.path.dirname(STK_EMB_DB_PATH), exist_ok=True)
    tmp_path = STK_EMB_DB_PATH + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump({'emb_model': model, 'embeddings': embeddings}, f, ensure_ascii=False)
    os.replace(tmp_path, STK_EMB_DB_PATH)


def _collect_sid_texts() -> tuple[list[tuple[int, str, str]], float]:
    mtime = os.path.getmtime(STK_DB_PATH)
    with open(STK_DB_PATH, 'r', encoding='utf-8') as f:
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
    return sid_texts, mtime


def _apply_sticker_cache(sid_texts: list[tuple[int, str, str]], emb_db: dict, mtime: float):
    global _sticker_cache, _sticker_cache_mtime
    cache = []
    for sid, text, path in sid_texts:
        k = f"{sid}:{text}"
        ek = f"e:{k}"
        if k in emb_db and ek in emb_db:
            cache.append((
                sid, text, path,
                np.array(emb_db[k], dtype=np.float32),
                np.array(emb_db[ek], dtype=np.float32),
            ))
    _sticker_cache = cache
    _sticker_cache_mtime = mtime


def _missing_indices(sid_texts: list[tuple[int, str, str]], emb_db: dict) -> list[int]:
    keys = [f"{sid}:{text}" for sid, text, _ in sid_texts]
    return [i for i, k in enumerate(keys) if k not in emb_db or f"e:{k}" not in emb_db]


async def _fill_missing_embeddings(model: str, sid_texts: list[tuple[int, str, str]], emb_db: dict) -> dict:
    caption_texts = [t for _, t, _ in sid_texts]
    emotion_texts = [t.split(',')[0] for t in caption_texts]
    keys = [f"{sid}:{text}" for sid, text, _ in sid_texts]
    missing_indices = _missing_indices(sid_texts, emb_db)
    if not missing_indices:
        return emb_db
    success_count = 0
    for batch_start in range(0, len(missing_indices), 10):
        batch_idx = missing_indices[batch_start:batch_start + 10]
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
    return emb_db


def _commit_sticker_cache(model: str, sid_texts: list[tuple[int, str, str]], emb_db: dict, mtime: float):
    keys = [f"{sid}:{text}" for sid, text, _ in sid_texts]
    used_keys = set(keys) | {f"e:{k}" for k in keys}
    new_emb_db = {k: v for k, v in emb_db.items() if k in used_keys}
    _apply_sticker_cache(sid_texts, new_emb_db, mtime)
    _save_emb_db(model, new_emb_db)
    info(f"Sticker缓存完成，共{len(_sticker_cache)}条向量")


async def _sticker_job_rebuild(model: str):
    info("开始后台重建表情包向量库")
    sid_texts, mtime = _collect_sid_texts()
    if not sid_texts:
        _apply_sticker_cache([], {}, mtime)
        return
    emb_db = await _fill_missing_embeddings(model, sid_texts, {})
    _commit_sticker_cache(model, sid_texts, emb_db, mtime)


async def _sticker_job_fill(model: str):
    sid_texts, mtime = _collect_sid_texts()
    ok, emb_db = _emb_db_usable(model)
    if not ok:
        emb_db = {}
    if not sid_texts:
        _apply_sticker_cache([], {}, mtime)
        return
    if not _missing_indices(sid_texts, emb_db):
        return
    info("开始后台补全缺失表情包向量")
    emb_db = await _fill_missing_embeddings(model, sid_texts, emb_db)
    _commit_sticker_cache(model, sid_texts, emb_db, mtime)


def _spawn_sticker_job(coro, name: str):
    global _sticker_job
    if _sticker_job_running():
        return

    async def runner():
        try:
            await coro
        except Exception as e:
            warning(f"{name}失败: {get_exc_desc(e)}")

    _sticker_job = asyncio.create_task(runner())


def _load_memory_from_disk(model: str) -> bool:
    """磁盘向量可用则装进内存。只在内存还空时读盘，之后以内存为准。"""
    ok, emb_db = _emb_db_usable(model)
    if not ok:
        return False
    if _sticker_cache:
        return True
    try:
        sid_texts, mtime = _collect_sid_texts()
    except FileNotFoundError:
        return False
    _apply_sticker_cache(sid_texts, emb_db, mtime)
    info(f"已从磁盘加载表情包向量，共{len(_sticker_cache)}条")
    return True


async def prepare_sticker_search() -> bool:
    """本轮能否拿 query 去搜表情包。必要时拉起后台建库，不阻塞 timeout。"""
    if _sticker_job_running():
        info("表情包向量任务进行中，本轮跳过发送")
        return False
    model = config.get('chat.sticker.emb_model')
    if not os.path.exists(STK_DB_PATH):
        return False
    if _load_memory_from_disk(model):
        return True
    info("表情包向量库缺失或模型不匹配，后台重建，本轮跳过发送")
    _spawn_sticker_job(_sticker_job_rebuild(model), "表情包向量库重建")
    return False


def schedule_sticker_fill():
    if _sticker_job_running():
        return
    model = config.get('chat.sticker.emb_model')
    ok, emb_db = _emb_db_usable(model)
    if not ok:
        return
    try:
        sid_texts, _ = _collect_sid_texts()
    except FileNotFoundError:
        return
    if not _missing_indices(sid_texts, emb_db):
        return
    _spawn_sticker_job(_sticker_job_fill(model), "表情包向量补全")

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

async def search_sticker(group_id: int, query: dict) -> tuple[str, int] | tuple[None, None]:
    threshold = config.get('chat.sticker.similarity_threshold')
    model = config.get('chat.sticker.emb_model')
    sticker_emotion = query.get('emotion', '')
    sticker_scene = query.get('scene', '')
    query = f"{sticker_emotion},{sticker_scene}" if sticker_emotion else sticker_scene
    try:
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

def _is_poke_msg(msg: Message) -> bool:
    return any(seg.get('type') == 'poke' for seg in msg.msg)


def _poke_target_id(msg: Message) -> int:
    for seg in msg.msg:
        if seg.get('type') == 'poke':
            try:
                return int(seg.get('data', {}).get('target_id') or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def _poke_key(msg: Message) -> tuple[int, int, int]:
    ts = int(msg.time.timestamp()) if isinstance(msg.time, datetime) else int(msg.time)
    return (ts, int(msg.user_id), _poke_target_id(msg))


group_pokes: dict[int, list[Message]] = {}
POKE_KEEP = 10


def remember_poke(msg: Message):
    lst = group_pokes.setdefault(msg.group_id, [])
    key = _poke_key(msg)
    if any(_poke_key(p) == key for p in lst):
        return
    lst.append(msg)
    del lst[:-POKE_KEEP]


def collect_recent_pokes(group_id: int, since: datetime) -> list[Message]:
    return [p for p in group_pokes.get(group_id, []) if p.time >= since]


def _poke_person_label(uid: int, name: str, self_id: int) -> str:
    if int(uid) == int(self_id):
        return "你"
    label = name or str(uid)
    return f"{label}({uid})"


async def format_msgs(
    msgs: list[Message], 
    image_caption_limit: int, 
    image_caption_prob: float,
    emotion_caption_limit: int,
    emotion_caption_prob: float,
    self_id: int = 0,
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
                case "poke":
                    from_label = _poke_person_label(msg.user_id, msg.nickname, self_id)
                    tid = int(sdata.get('target_id') or 0)
                    tname = sdata.get('target_name') or str(tid)
                    to_label = _poke_person_label(tid, tname, self_id)
                    text = f"{get_readable_datetime(msg.time)} {from_label} 戳了戳 {to_label}"
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

    is_poke = _is_poke_msg(msg)
    poke_target = _poke_target_id(msg) if is_poke else 0
    if is_poke:
        remember_poke(msg)

    if msg.user_id == self_id:  # 自己发的消息/自己戳人不触发
        return
    if not is_poke and get_plain_text(msg).startswith("/"):  # 命令消息不触发
        return
    if is_poke and poke_target != self_id:  # 别人戳别人，只入时间线
        return

    status = GroupStatus.load(msg.group_id)
    # 忽略上次回复思考时接收到的消息
    if status.last_reply_time and msg.time.timestamp() <= status.last_reply_time:
        return
    
    if is_poke:
        info(f"{msg.group_id} 的戳一戳 {msg.nickname}({msg.user_id}) -> {poke_target}")
    else:
        info(f"{msg.group_id} 的新消息 {msg.msg_id} {msg.nickname}({msg.user_id}): {get_plain_text(msg)}")
    
    # ---------------- 更新意愿值 ---------------- #

    try:
        delta = 0.0
        # 随时间减少
        if status.last_check_willing_time:
            time_passed = time.time() - status.last_check_willing_time
            delta -= min(config.get('chat.willing.decrease_per_minute') * time_passed / 60.0, status.willingness)
        if is_poke:
            delta += config.get('chat.willing.increase_per_poke')
        else:
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
        if not is_poke and msg.msg_id and not any(m.msg_id == msg.msg_id for m in recent_msgs):
            recent_msgs.append(msg)
        since = min((m.time for m in recent_msgs), default=msg.time)
        poke_keys = {_poke_key(m) for m in recent_msgs if _is_poke_msg(m)}
        for poke in collect_recent_pokes(msg.group_id, since):
            key = _poke_key(poke)
            if key not in poke_keys:
                recent_msgs.append(poke)
                poke_keys.add(key)
        info(f"获取最近共 {len(recent_msgs)} 条有效聊天记录")

        recent_text = await format_msgs(
            recent_msgs,
            image_caption_limit=config.get('image_caption.image_limit'),
            image_caption_prob=config.get('image_caption.image_prob'),
            emotion_caption_limit=config.get('image_caption.emotion_limit'),
            emotion_caption_prob=config.get('image_caption.emotion_prob'),
            self_id=self_id,
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
                    if sm.sticker:
                        body = f"[表情包: {sm.sticker}]"
                    else:
                        body = sm.text
                    sm_text += f"{get_readable_datetime(sm.time)} [{sm.id}]: {body}\n"
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
                    { 'key': 'actions', 'type': 'list' },
                    { 'key': 'user_updates', 'type': 'list' },
                ],
            }
        )
        info(f"LLM生成回复成功: {llm_response}")

        actions = _parse_actions(llm_response)
        user_updates = llm_response.get('user_updates', [])

    except:
        error(f"请求LLM生成回复时失败，放弃聊天处理")
        return

    # ---------------- 发送动作 ---------------- #

    try:
        send_msg_id_texts: list[tuple[int, str, str]] = []
        first_action = True

        async def wait_interval():
            nonlocal first_action
            if first_action:
                first_action = False
                return
            await asyncio.sleep(config.get('chat.reply_interval_seconds'))

        async def note_sent_msg(send_msg_id: int):
            status.load(msg.group_id)
            status.self_msg_ids.append(send_msg_id)
            status.self_msg_ids = status.self_msg_ids[-100:]
            status.last_reply_time = time.time()
            status.save()

        async def process_reply_text(index: int, text: str):
            if not text:
                info(f"LLM生成的回复{index}为空，放弃发送")
                return
            at_id, reply_id = None, None
            if at_match := re.search(r"\[@(\d+)\]", text):
                at_id = int(at_match.group(1))
                text = text.replace(at_match.group(0), "")
                if any(m.user_id == at_id for m in recent_msgs):
                    text = f"[CQ:at,qq={at_id}]" + text
            if reply_match := re.search(r"\[reply=(-?\d+)\]", text):
                reply_id = int(reply_match.group(1))
                text = text.replace(reply_match.group(0), "")
                if any(int(m.msg_id) == reply_id for m in recent_msgs):
                    text = f"[CQ:reply,id={reply_id}]" + text
            text = truncate(text, config.get('chat.reply_max_length'))
            info(f"自动聊天生成回复{index}: {text} at_id={at_id} reply_id={reply_id}")

            send_ret = await rpc_send_group_msg(msg.group_id, text)
            send_msg_id = int(send_ret['message_id'])
            send_msg_id_texts.append((send_msg_id, 'text', text))
            info(f"发送回复{index}成功: send_msg_id={send_msg_id}")
            await note_sent_msg(send_msg_id)

        async def process_sticker(hit: tuple, query: dict | None = None):
            sticker_path, sticker_sid, sticker_all_sids, sticker_old_multipliers = hit
            if not sticker_path:
                info("未匹配到表情包，跳过发送")
                return
            send_ret = await rpc_send_group_msg(msg.group_id, f"[CQ:image,file=file://{sticker_path}]")
            send_msg_id = int(send_ret['message_id'])
            await note_sent_msg(send_msg_id)
            info(f"表情包发送成功: sid={sticker_sid}")
            query = query or {}
            emotion = str(query.get('emotion') or '').strip()
            scene = str(query.get('scene') or '').strip()
            if emotion and scene:
                sticker_desc = f"{emotion}/{scene}"
            else:
                sticker_desc = emotion or scene or f"sid={sticker_sid}"
            send_msg_id_texts.append((send_msg_id, 'sticker', sticker_desc))
            update_sticker_multipliers(msg.group_id, sticker_sid, sticker_all_sids)
            changed = [f"sid={s}({sticker_old_multipliers[s]:.1f}→{get_sticker_multiplier(msg.group_id, s):.1f})"
                       for s in sticker_all_sids if abs(get_sticker_multiplier(msg.group_id, s) - sticker_old_multipliers[s]) > 0.001]
            if changed:
                info(f"表情包倍率: {', '.join(changed)}")

        async def process_poke(ids: list[int]):
            for uid in ids:
                try:
                    await rpc_poke_group_member(msg.group_id, uid)
                    info(f"戳一戳成功: user_id={uid}")
                except Exception as e:
                    warning(f"戳一戳失败 user_id={uid}: {get_exc_desc(e)}")

        async def process_react(msg_id: int, emoji_id: str):
            if not any(int(m.msg_id) == msg_id for m in recent_msgs):
                info(f"贴表情跳过，消息不在最近记录中: msg_id={msg_id}")
                return
            try:
                await rpc_set_msg_emoji_like(msg.group_id, msg_id, emoji_id)
                info(f"贴表情成功: msg_id={msg_id} emoji_id={emoji_id}")
            except Exception as e:
                warning(f"贴表情失败 msg_id={msg_id} emoji_id={emoji_id}: {get_exc_desc(e)}")

        sticker_hits: dict[int, tuple] = {}
        sticker_indexes = [i for i, a in enumerate(actions) if a['kind'] == 'sticker']
        if sticker_indexes:
            can_search = await prepare_sticker_search()
            if can_search:
                async def prefetch_stickers():
                    for i in sticker_indexes:
                        sticker_hits[i] = await search_sticker(msg.group_id, actions[i]['query'])

                try:
                    await asyncio.wait_for(
                        prefetch_stickers(),
                        timeout=float(config.get('chat.sticker.timeout')),
                    )
                except asyncio.TimeoutError:
                    warning("Sticker搜索超时，抛弃未就绪的表情包")
                except BaseException as e:
                    warning(f"Sticker搜索失败: {get_exc_desc(e)}")
                schedule_sticker_fill()

        exec_actions = []
        for i, action in enumerate(actions):
            if action['kind'] != 'sticker':
                exec_actions.append(action)
                continue
            hit = sticker_hits.get(i)
            if not hit or not hit[0]:
                if i not in sticker_hits:
                    info("表情包未就绪，跳过发送")
                else:
                    info("未匹配到表情包，跳过发送")
                continue
            exec_actions.append({**action, 'hit': hit})

        text_index = 0
        for action in exec_actions:
            kind = action['kind']
            await wait_interval()
            if kind == 'text':
                text_index += 1
                await process_reply_text(text_index, action['text'])
            elif kind == 'sticker':
                await process_sticker(action['hit'], action.get('query'))
            elif kind == 'poke':
                await process_poke(action['ids'])
            elif kind == 'react':
                await process_react(action['msg_id'], action['emoji_id'])

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
        keep_count = config.get('chat.mem.sm_keep_count')
        for msg_id, kind, content in send_msg_id_texts:
            if kind == 'sticker':
                mem.sm_add(msg_id=msg_id, keep_count=keep_count, sticker=content)
            else:
                mem.sm_add(msg_id=msg_id, keep_count=keep_count, text=content)

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