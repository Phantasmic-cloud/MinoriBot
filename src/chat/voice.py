import asyncio
import base64
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime

from src.utils import *
from .autochat import chat_gwl
from src.pjsk.asset import (
    RegionMasterDataCollection,
    RegionRipAssetManger,
    archive_voice_path,
    live2d_voice_path,
)
config = Config("chat.voice")
logger = get_logger("chat")
file_db = get_file_db("data/chat/autochat/voice_db.json")
cd = ColdDown(file_db, logger)

VOICE_DIR = "data/chat/autochat/voice"
VOICE_INBOX_DIR = "data/chat/autochat/voice_inbox"
VOICE_INBOX_DONE_DIR = os.path.join(VOICE_INBOX_DIR, "_done")
AUDIO_EXTS = {".mp3", ".wav", ".amr", ".silk", ".slk", ".ogg", ".m4a", ".aac", ".flac", ".wma"}
TAG_MAX_LEN = 64


# ======================= 逻辑处理 ======================= #


@dataclass
class VoiceClip:
    """语音条目：原文件和台词 tag。"""
    vid: int
    path: str
    tag: str
    source: str = ""
    key: str = ""


class VoiceManager:
    """语音仓库：加载、增删、改 tag。"""
    _mgr: "VoiceManager" = None

    def __init__(self):
        self.vid_top = 0
        self.voices: dict[int, VoiceClip] = {}

    def _load(self):
        self.vid_top = file_db.get("vid_top", 0)
        self.voices = {}
        for vid_str, v in file_db.get("voices", {}).items():
            self.voices[int(vid_str)] = VoiceClip(
                vid=int(v["vid"]),
                path=v["path"],
                tag=str(v.get("tag") or ""),
                source=str(v.get("source") or ""),
                key=str(v.get("key") or ""),
            )
        logger.info("成功加载%s条语音, vid_top=%s", len(self.voices), self.vid_top)

    def _save(self):
        file_db.set("vid_top", self.vid_top)
        file_db.set("voices", {str(vid): asdict(v) for vid, v in self.voices.items()})

    @classmethod
    def get(cls) -> "VoiceManager":
        if cls._mgr is None:
            cls._mgr = VoiceManager()
            cls._mgr._load()
        return cls._mgr

    def find(self, vid: int) -> VoiceClip | None:
        return self.voices.get(int(vid))

    def find_by_tag(self, tag: str) -> VoiceClip | None:
        tag = _normalize_tag(tag)
        if not tag:
            return None
        for v in self.voices.values():
            if v.tag == tag:
                return v
        return None

    def find_by_key(self, source: str, key: str) -> VoiceClip | None:
        if not source or not key:
            return None
        for v in self.voices.values():
            if v.source == source and v.key == key:
                return v
        return None

    def add(self, src_path: str | None, tag: str, source: str = "", key: str = "") -> int:
        tag = _normalize_tag(tag)
        assert tag, "tag 不能为空"
        self.vid_top += 1
        vid = self.vid_top
        ext = ".mp3"
        if src_path:
            _, ext = os.path.splitext(src_path)
            ext = (ext or ".mp3").lower()
            if ext not in AUDIO_EXTS:
                ext = ".mp3"
        os.makedirs(VOICE_DIR, exist_ok=True)
        dst = os.path.join(VOICE_DIR, f"{vid}{ext}")
        if src_path and os.path.isfile(src_path):
            shutil.copy2(src_path, dst)
        self.voices[vid] = VoiceClip(vid=vid, path=dst, tag=tag, source=source, key=key)
        self._save()
        return vid

    def set_tag(self, vid: int, tag: str):
        v = self.voices.get(int(vid))
        assert v is not None, f"语音vid={vid}不存在"
        tag = _normalize_tag(tag)
        assert tag, "tag 不能为空"
        v.tag = tag
        self._save()

    def delete(self, vid: int):
        v = self.voices.get(int(vid))
        assert v is not None, f"语音vid={vid}不存在"
        try:
            if os.path.exists(v.path):
                os.remove(v.path)
        except Exception as e:
            logger.warning("删除语音vid=%s文件失败: %s", vid, get_exc_desc(e))
        del self.voices[int(vid)]
        self._save()

    def all(self) -> list[VoiceClip]:
        return sorted(self.voices.values(), key=lambda v: v.vid)


def _normalize_tag(tag: str) -> str:
    return truncate(" ".join(str(tag or "").split()), TAG_MAX_LEN).strip()


def _tag_from_filename(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    return _normalize_tag(base)


def _guess_ext(data: dict) -> str:
    for key in ("name", "file", "path", "url"):
        raw = str(data.get(key) or "").split("?", 1)[0]
        ext = os.path.splitext(raw)[1].lower()
        if ext in AUDIO_EXTS:
            return ext[1:]
    return "mp3"


def _extract_audio_items(msg) -> list[dict]:
    if not msg:
        return []
    cqs = extract_cq_code(msg)
    items: list[dict] = []
    for d in cqs.get("record", []):
        items.append({"kind": "record", **d})
    for d in cqs.get("file", []):
        name = str(d.get("name") or d.get("file") or "")
        ext = os.path.splitext(name)[1].lower()
        if ext in AUDIO_EXTS:
            items.append({"kind": "file", **d})
    return items


def _local_path(raw) -> str | None:
    if not raw:
        return None
    path = str(raw)
    if path.startswith("file://"):
        path = path[7:]
    if os.path.isfile(path):
        return path
    return None


async def _copy_from_get_record(bot, file_id: str, dest: str) -> bool:
    ret = await bot.get_record(str(file_id), out_format="mp3")
    if not isinstance(ret, dict):
        return False
    if ret.get("base64"):
        with open(dest, "wb") as f:
            f.write(base64.b64decode(ret["base64"]))
        return True
    p = ret.get("file") or ret.get("path") or ""
    local = _local_path(p)
    if local:
        shutil.copy2(local, dest)
        return True
    if str(p).startswith("http"):
        await download_file(p, dest)
        return True
    return False


async def _materialize_audio(bot, data: dict, dest: str):
    """把消息里的语音/音频文件落到 dest。"""
    file_id = str(data.get("file") or data.get("file_id") or "")
    url = str(data.get("url") or "")
    path = str(data.get("path") or "")
    name = str(data.get("name") or "")

    if data.get("kind") == "record":
        last_err = None
        for fid in (file_id, url, name):
            if not fid:
                continue
            try:
                if await _copy_from_get_record(bot, fid, dest):
                    return
            except Exception as e:
                last_err = e
        raise ReplyException(
            f"下载语音失败: {get_exc_desc(last_err) if last_err else 'get_record 没有返回文件'}"
        )

    for candidate in (path, file_id, url, name):
        local = _local_path(candidate)
        if local:
            shutil.copy2(local, dest)
            return
        if str(candidate).startswith("http"):
            try:
                await download_file(candidate, dest)
                return
            except Exception:
                pass

    last_err = None
    for fid in (file_id, url, name):
        if not fid:
            continue
        try:
            if await _copy_from_get_record(bot, fid, dest):
                return
        except Exception as e:
            last_err = e
    raise ReplyException(
        f"下载语音失败: {get_exc_desc(last_err) if last_err else '消息里没有可用的语音文件'}"
    )


async def _pick_audio_item(ctx: HandlerContext) -> dict:
    items = _extract_audio_items(ctx.get_msg())
    if items:
        return items[0]
    reply_msg = await ctx.aget_reply_msg()
    items = _extract_audio_items(reply_msg)
    if items:
        return items[0]
    raise ReplyException("请回复一条语音，或在消息里附带语音/音频文件")


def _usage() -> str:
    return (
        "使用方式:\n"
        "/voice upload 台词  （回复语音或附带语音）\n"
        "/voice {vid}\n"
        "/voice tag {vid} 新台词\n"
        "/voice del {vid}\n"
        "/voice update\n"
        "/all voice"
    )


def _clip_line(v: VoiceClip) -> str:
    return f"vid={v.vid}  {v.tag}"


def _serif_to_tag(serif: str) -> str:
    text = " ".join(str(serif or "").replace("\r", " ").replace("\n", " ").split())
    return _normalize_tag(text)


_pjsk_queue: asyncio.Queue | None = None
_pjsk_queued: set[str] = set()
_pjsk_worker_started = False


class _PjskDownloadBatch:
    def __init__(self, bot, event):
        self.bot = bot
        self.event = event
        self.pending: set[str] = set()
        self.ok = 0
        self.fail = 0


def _pjsk_queue_get() -> asyncio.Queue:
    global _pjsk_queue
    if _pjsk_queue is None:
        _pjsk_queue = asyncio.Queue()
    return _pjsk_queue


def _pjsk_region() -> str:
    return str(config.get("pjsk.region", "cn") or "cn")


async def _pjsk_download_one(vid: int, asset_path: str):
    mgr = VoiceManager.get()
    clip = mgr.find(vid)
    if clip is None:
        return
    if os.path.isfile(clip.path):
        return
    rip = RegionRipAssetManger.get(_pjsk_region())
    src = await rip.get_asset_cache_path(asset_path, allow_error=False)
    os.makedirs(os.path.dirname(clip.path) or ".", exist_ok=True)
    shutil.copy2(src, clip.path)
    logger.info("语音下载完成 vid=%s %s", vid, clip.tag)


async def _notify_pjsk_batch(batch: _PjskDownloadBatch):
    try:
        await send_reply_msg(batch.bot, batch.event, f"pjsk 下载完成: 成功{batch.ok} 失败{batch.fail}")
    except Exception:
        logger.print_exc("语音下载完成通知失败")


async def _pjsk_download_worker():
    interval = float(config.get("pjsk.download_interval_seconds", 3) or 3)
    queue = _pjsk_queue_get()
    while True:
        vid, asset_path, key, batch = await queue.get()
        try:
            await _pjsk_download_one(vid, asset_path)
            if batch is not None:
                batch.ok += 1
        except Exception:
            logger.print_exc(f"语音下载失败 vid={vid}")
            if batch is not None:
                batch.fail += 1
        finally:
            _pjsk_queued.discard(key)
            if batch is not None:
                batch.pending.discard(key)
                if not batch.pending:
                    await _notify_pjsk_batch(batch)
            queue.task_done()
        await asyncio.sleep(max(interval, 0.5))


def _ensure_pjsk_worker():
    global _pjsk_worker_started
    if _pjsk_worker_started:
        return
    try:
        asyncio.get_running_loop().create_task(_pjsk_download_worker())
        _pjsk_worker_started = True
    except RuntimeError:
        pass


def _enqueue_pjsk_download(vid: int, asset_path: str, key: str, batch: _PjskDownloadBatch | None = None):
    if key in _pjsk_queued:
        return False
    _pjsk_queued.add(key)
    if batch is not None:
        batch.pending.add(key)
    _ensure_pjsk_worker()
    _pjsk_queue_get().put_nowait((vid, asset_path, key, batch))
    return True


_PJSK_ARCHIVE_VOICE_TYPES = {"practice"}


async def _pjsk_character_name_en(md: RegionMasterDataCollection, cid: int) -> str:
    row = await md.game_characters.find_by_id(cid)
    assert_and_reply(row, f"找不到角色 id={cid}")
    name = str(row.get("givenNameEnglish") or "").strip()
    assert_and_reply(name, f"角色 id={cid} 没有英文名")
    return name


async def _collect_pjsk_live2d_clips(md: RegionMasterDataCollection, cid: int) -> list[tuple[str, str, str]]:
    rows = await md.system_live2ds.find_by("characterId", cid, mode="all")
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for row in rows or []:
        voice = str(row.get("voice") or "").strip()
        bundle = str(row.get("assetbundleName") or "").strip()
        tag = _serif_to_tag(row.get("serif") or "")
        if not voice or not bundle or not tag:
            continue
        key = f"live2d:{bundle}/{voice}"
        if key in seen:
            continue
        seen.add(key)
        out.append((key, live2d_voice_path(bundle, voice), tag))
    return out


async def _collect_pjsk_archive_clips(md: RegionMasterDataCollection, cid: int) -> list[tuple[str, str, str]]:
    name_en = await _pjsk_character_name_en(md, cid)
    rows = await md.character_archive_voices.find_by("gameCharacterId", cid, mode="all")
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for row in rows or []:
        voice_type = str(row.get("characterArchiveVoiceType") or "").strip()
        if voice_type not in _PJSK_ARCHIVE_VOICE_TYPES:
            continue
        asset_name = str(row.get("assetName") or "").strip()
        tag = _serif_to_tag(row.get("displayPhrase") or "")
        if not asset_name or not tag:
            continue
        key = f"archive:{voice_type}/{asset_name}"
        if key in seen:
            continue
        seen.add(key)
        out.append((key, archive_voice_path(cid, name_en, asset_name), tag))
    return out


async def _collect_pjsk_clips() -> list[tuple[str, str, str]]:
    cid = int(config.get("pjsk.character_id", 5) or 5)
    md = RegionMasterDataCollection(_pjsk_region())
    clips = await _collect_pjsk_live2d_clips(md, cid)
    clips.extend(await _collect_pjsk_archive_clips(md, cid))
    return clips


async def _scan_inbox(mgr: VoiceManager) -> tuple[list[str], list[str], list[str]]:
    os.makedirs(VOICE_INBOX_DIR, exist_ok=True)
    os.makedirs(VOICE_INBOX_DONE_DIR, exist_ok=True)
    names = sorted(
        n for n in os.listdir(VOICE_INBOX_DIR)
        if not n.startswith(".") and n != "_done"
    )
    added, skipped, failed = [], [], []
    for name in names:
        src = os.path.join(VOICE_INBOX_DIR, name)
        if not os.path.isfile(src):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in AUDIO_EXTS:
            skipped.append(f"{name}(非音频)")
            continue
        tag = _tag_from_filename(name)
        if not tag:
            skipped.append(f"{name}(无tag)")
            continue
        if mgr.find_by_tag(tag) is not None:
            skipped.append(f"{name}(tag已存在)")
            continue
        try:
            vid = mgr.add(src, tag, source="inbox", key=name)
            dst_done = os.path.join(VOICE_INBOX_DONE_DIR, name)
            if os.path.exists(dst_done):
                stem, e = os.path.splitext(name)
                dst_done = os.path.join(VOICE_INBOX_DONE_DIR, f"{stem}_{int(datetime.now().timestamp())}{e}")
            shutil.move(src, dst_done)
            added.append(f"vid={vid} {tag}")
        except Exception as e:
            logger.print_exc(f"扫描导入语音失败: {name}")
            failed.append(f"{name}({get_exc_desc(e)})")
    return added, skipped, failed


async def _scan_pjsk(mgr: VoiceManager) -> tuple[int, int, int, list[tuple[int, str, str]]]:
    clips = await _collect_pjsk_clips()
    added = existed = 0
    jobs: list[tuple[int, str, str]] = []
    for key, asset_path, tag in clips:
        clip = mgr.find_by_key("pjsk", key)
        if clip is None:
            dup = mgr.find_by_tag(tag)
            if dup is not None:
                existed += 1
                continue
            vid = mgr.add(None, tag, source="pjsk", key=key)
            clip = mgr.find(vid)
            added += 1
        else:
            existed += 1
        if clip and os.path.isfile(clip.path):
            continue
        if clip:
            jobs.append((clip.vid, asset_path, key))
    return added, len(jobs), existed, jobs


# ======================= 指令处理 ======================= #


voice_upload = CmdHandler(["/voice upload", "/语音上传"], logger)
voice_upload.check_cdrate(cd).check_wblist(chat_gwl)


@voice_upload.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()
    force = False
    if args.endswith(" force"):
        force = True
        args = args[: -len(" force")].strip()
    elif args.startswith("force "):
        force = True
        args = args[len("force "):].strip()
    tag = _normalize_tag(args)
    assert_and_reply(tag, "请填写台词 tag，例如: /voice upload 那我今晚真不去")

    mgr = VoiceManager.get()
    if not force:
        dup = mgr.find_by_tag(tag)
        if dup is not None:
            raise ReplyException(f"已有相同 tag 的语音(vid={dup.vid})，使用 \"/voice upload force 台词\" 强制上传")

    item = await _pick_audio_item(ctx)
    ext = "mp3" if item.get("kind") == "record" else _guess_ext(item)
    with TempFilePath(ext) as tmp_path:
        await _materialize_audio(ctx.bot, item, tmp_path)
        vid = mgr.add(tmp_path, tag)
    await ctx.asend_reply_msg(f"上传成功 {_clip_line(mgr.find(vid))}")


voice_tag = CmdHandler(["/voice tag", "/语音标签"], logger)
voice_tag.check_cdrate(cd).check_wblist(chat_gwl)


@voice_tag.handle()
async def _(ctx: HandlerContext):
    parts = ctx.get_args().strip().split(None, 1)
    try:
        vid = int(parts[0])
        tag = _normalize_tag(parts[1] if len(parts) > 1 else "")
    except Exception:
        raise ReplyException("使用方式: /voice tag {vid} 新台词")
    assert_and_reply(tag, "使用方式: /voice tag {vid} 新台词")
    VoiceManager.get().set_tag(vid, tag)
    await ctx.asend_reply_msg(f"已更新 {_clip_line(VoiceManager.get().find(vid))}")


voice_del = CmdHandler(["/voice del", "/语音删除"], logger)
voice_del.check_cdrate(cd).check_wblist(chat_gwl)


@voice_del.handle()
async def _(ctx: HandlerContext):
    try:
        vid = int(ctx.get_args().strip())
    except Exception:
        raise ReplyException("使用方式: /voice del {vid}")
    VoiceManager.get().delete(vid)
    await ctx.asend_reply_msg(f"已删除语音 vid={vid}")


voice_update = CmdHandler(["/voice update", "/更新语音", "/voice scan", "/语音扫描"], logger)
voice_update.check_cdrate(cd).check_wblist(chat_gwl)


@voice_update.handle()
async def _(ctx: HandlerContext):
    mgr = VoiceManager.get()
    added, skipped, failed = await _scan_inbox(mgr)
    pjsk_added = pjsk_queued = pjsk_existed = 0
    jobs: list[tuple[int, str, str]] = []
    try:
        pjsk_added, pjsk_queued, pjsk_existed, jobs = await _scan_pjsk(mgr)
    except Exception as e:
        logger.print_exc("扫描 pjsk 语音失败")
        failed.append(f"pjsk({get_exc_desc(e)})")

    batch = _PjskDownloadBatch(ctx.bot, ctx.event)
    queued = 0
    for vid, asset_path, key in jobs:
        if _enqueue_pjsk_download(vid, asset_path, key, batch):
            queued += 1
    pjsk_queued = queued

    lines = [
        f"inbox: 导入{len(added)} 跳过{len(skipped)} 失败{len(failed)}",
        f"pjsk: 新增{pjsk_added} 排队下载{pjsk_queued} 已有{pjsk_existed}",
    ]
    if added:
        lines.append("导入:\n" + "\n".join(added))
    if skipped:
        lines.append("跳过:\n" + "\n".join(skipped))
    if failed:
        lines.append("失败:\n" + "\n".join(failed))
    await ctx.asend_fold_msg_adaptive("\n".join(lines))


voice_all = CmdHandler(["/all voice", "/所有语音"], logger)
voice_all.check_cdrate(cd).check_wblist(chat_gwl)


@voice_all.handle()
async def _(ctx: HandlerContext):
    voices = VoiceManager.get().all()
    if not voices:
        raise ReplyException("当前没有任何语音")
    msg = f"共{len(voices)}条语音\n" + "\n".join(_clip_line(v) for v in voices)
    await ctx.asend_fold_msg_adaptive(msg)


voice_view = CmdHandler(["/voice", "/语音"], logger)
voice_view.check_cdrate(cd).check_wblist(chat_gwl)


@voice_view.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()
    if not args:
        raise ReplyException(_usage())
    try:
        vid = int(args)
    except Exception:
        raise ReplyException(_usage())
    v = VoiceManager.get().find(vid)
    if v is None:
        raise ReplyException(f"语音vid={vid}不存在")
    if not os.path.isfile(v.path):
        raise ReplyException("语音文件还没下好")
    await ctx.asend_msg(f"[CQ:record,file=file://{os.path.abspath(v.path)}]")
    await ctx.asend_reply_msg(v.tag)
