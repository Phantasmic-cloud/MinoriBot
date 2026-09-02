import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
from PIL import Image

from src.llm import ChatSession
from src.utils import *

config = Config("chat.sticker")
logger = get_logger("chat")
file_db = get_file_db("data/chat/autochat/sticker_db.json")
cd = ColdDown(file_db, logger)

STICKER_DIR = "data/chat/autochat/sticker"
STICKER_THUMB_SIZE = (80, 80)
STICKER_THUMB_BG = (230, 240, 255, 255)


# ======================= 逻辑处理 ======================= #


@dataclass
class Sticker:
    """表情包条目：原图、理解文案、缩略图和查重哈希。"""
    sid: int
    path: str
    caption: list
    thumb_path: str | None = None
    hash1: str | None = None
    hash2: str | None = None

    def caption_texts(self) -> list[str]:
        """拼出 emotion,scene 文案列表。"""
        return [f"{c['emotion']},{c['scene']}" for c in self.caption if c.get("emotion") and c.get("scene")]

    @classmethod
    def load(cls, data: dict) -> "Sticker":
        raw_caption = data.get("caption", [])
        if isinstance(raw_caption, str):
            raw_caption = [{"emotion": "", "scene": raw_caption}] if raw_caption else []
        return cls(
            sid=data["sid"],
            path=data["path"],
            caption=raw_caption,
            thumb_path=data.get("thumb_path", None),
            hash1=data.get("hash1", None),
            hash2=data.get("hash2", None),
        )

    def calc_hash(self):
        """算感知哈希，用来查重。"""
        image = Image.open(self.path)
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            image = image.convert("RGBA").resize((64, 64), Image.Resampling.BILINEAR)
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            bg.alpha_composite(image)
            image = bg
        image = image.convert("RGB")
        image = image.resize((16, 16), Image.Resampling.BILINEAR).convert("L")
        self.hash2 = image.tobytes().hex()
        image = image.resize((8, 8), Image.Resampling.BILINEAR)
        pixels = np.array(image).flatten()
        avg = pixels.mean()
        bits = 0
        for idx, p in enumerate(pixels):
            if p >= avg:
                bits |= 1 << (63 - idx)
        self.hash1 = f"{bits:016x}"

    def is_same(self, other: "Sticker") -> bool:
        """两张图是否判为重复。"""
        if not self.hash1 or not other.hash1:
            return False
        h1_thresh = config.get("duplicate.hash1_threshold")
        h2_thresh = config.get("duplicate.hash2_threshold")
        if (int(self.hash1, 16) ^ int(other.hash1, 16)).bit_count() > h1_thresh:
            return False
        img1 = np.frombuffer(bytes.fromhex(self.hash2), dtype=np.uint8)
        img2 = np.frombuffer(bytes.fromhex(other.hash2), dtype=np.uint8)
        diff = np.sum(np.abs(img1.astype(np.int16) - img2.astype(np.int16)))
        return diff <= h2_thresh

    def ensure_thumb(self):
        """没有缩略图就现生成一张。"""
        try:
            if self.thumb_path is None:
                name = os.path.basename(self.path)
                self.thumb_path = os.path.join(STICKER_DIR, f"{name}_thumb.jpg")
            if not os.path.exists(self.thumb_path):
                img = Image.open(self.path).convert("RGBA")
                img.thumbnail(STICKER_THUMB_SIZE, Image.Resampling.LANCZOS)
                bg = Image.new("RGBA", img.size, STICKER_THUMB_BG)
                bg.alpha_composite(img)
                bg.convert("RGB").save(self.thumb_path, format="JPEG", quality=85)
        except Exception as e:
            logger.warning("生成表情包sid=%s缩略图失败: %s", self.sid, e)
            self.thumb_path = None


class StickerManager:
    """表情包仓库：加载、查重、增删。"""
    _mgr: "StickerManager" = None

    def __init__(self):
        self.sid_top = 0
        self.stickers: dict[int, Sticker] = {}

    def _load(self):
        """从文件库载入全部表情包。"""
        self.sid_top = file_db.get("sid_top", 0)
        self.stickers = {}
        for sid_str, s in file_db.get("stickers", {}).items():
            self.stickers[int(sid_str)] = Sticker.load(s)
        logger.info("成功加载%s个表情包, sid_top=%s", len(self.stickers), self.sid_top)

    def _save(self):
        """把当前表情包写回文件库。"""
        file_db.set("sid_top", self.sid_top)
        file_db.set("stickers", {str(sid): asdict(s) for sid, s in self.stickers.items()})

    @classmethod
    def get(cls) -> "StickerManager":
        if cls._mgr is None:
            cls._mgr = StickerManager()
            cls._mgr._load()
        return cls._mgr

    def find(self, sid: int) -> Sticker | None:
        """按 sid 查找表情包。"""
        return self.stickers.get(sid)

    def check_duplicate(self, img_path: str) -> int | None:
        """查重，重复则返回已有 sid。"""
        tmp = Sticker(sid=0, path=img_path, caption="")
        tmp.calc_hash()
        dirty = False
        for s in self.stickers.values():
            if s.hash1 is None:
                try:
                    s.calc_hash()
                    dirty = True
                except Exception as e:
                    logger.warning("补算表情包sid=%s hash失败: %s", s.sid, e)
                    continue
            if s.is_same(tmp):
                if dirty:
                    self._save()
                return s.sid
        if dirty:
            self._save()
        return None

    def add(self, img_path: str, caption) -> int:
        """拷贝图片入库，返回新 sid。"""
        self.sid_top += 1
        sid = self.sid_top
        _, ext = os.path.splitext(img_path)
        os.makedirs(STICKER_DIR, exist_ok=True)
        dst = os.path.join(STICKER_DIR, f"{sid}{ext}")
        shutil.copy2(img_path, dst)
        s = Sticker(sid=sid, path=dst, caption=caption)
        s.calc_hash()
        self.stickers[sid] = s
        self._save()
        return sid

    def delete(self, sid: int):
        """按 sid 删掉表情包和文件。"""
        s = self.stickers.get(sid)
        assert s is not None, f"表情包sid={sid}不存在"
        try:
            if os.path.exists(s.path):
                os.remove(s.path)
            if s.thumb_path and os.path.exists(s.thumb_path):
                os.remove(s.thumb_path)
        except Exception as e:
            logger.warning("删除表情包sid=%s文件失败: %s", sid, get_exc_desc(e))
        del self.stickers[sid]
        self._save()

    def all(self) -> list[Sticker]:
        """按 sid 排序返回全部表情包。"""
        return sorted(self.stickers.values(), key=lambda s: s.sid)


async def generate_caption(img_path: str):
    """用 LLM 给表情包写 emotion/scene 理解。"""
    model = config.get("caption.model")
    timeout = config.get("caption.timeout")
    max_tokens = config.get("caption.max_tokens")
    prompt = config.get("caption.prompt")
    img = Image.open(img_path).convert("RGB")
    session = ChatSession()
    session.append_user_content(prompt, imgs=[img], verbose=False)

    t = datetime.now()
    resp = await session.get_response(model_name=model, timeout=timeout, max_tokens=max_tokens)
    cost_seconds = (datetime.now() - t).total_seconds()
    text = resp.result.strip() or (resp.reasoning or "").strip()
    assert text, "表情包理解为空"
    try:
        start = text.find("{")
        end = text.rfind("}")
        assert start != -1 and end != -1
        data = loads_json(text[start:end + 1])
        captions = [c for c in data.get("captions", []) if c.get("emotion") and c.get("scene")]
        assert captions
    except Exception as e:
        raise Exception(f"解析表情包理解json失败: {e}\n原始: {text[:200]}")
    model_name_str = resp.model.get_full_name()
    return captions, model_name_str, cost_seconds, resp.prompt_tokens, resp.completion_tokens


# ======================= 指令处理 ======================= #

sticker_upload = CmdHandler(["/sticker upload", "/stku", "/表情包上传"], logger)
sticker_upload.check_cdrate(cd)


@sticker_upload.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()
    force = "force" in args

    image_data = await ctx.aget_image_datas(return_first=True, max_count=1)
    if not image_data:
        raise ReplyException("请附带一张图片")

    async with TempDownloadFilePath(image_data["url"], "gif") as path:
        if not force:
            dup_sid = await run_in_pool(StickerManager.get().check_duplicate, path)
            if dup_sid is not None:
                raise ReplyException(f"与已有表情包重复(sid={dup_sid})，使用\"/sticker upload force\"强制上传")
        captions, model_str, cost_s, ptokens, ctokens = await generate_caption(path)
        sid = StickerManager.get().add(path, captions)

    lines = "\n".join([f"· {c['emotion']}，{c['scene']}" for c in captions])
    await ctx.asend_reply_msg(
        f"上传成功 sid={sid}\n{lines}\n"
        f"({model_str} | {cost_s:.1f}s, {ptokens}+{ctokens} tokens)"
    )


sticker_del = CmdHandler(["/sticker del", "/stk del", "/表情包删除"], logger)
sticker_del.check_cdrate(cd)


@sticker_del.handle()
async def _(ctx: HandlerContext):
    try:
        sid = int(ctx.get_args().strip())
    except Exception:
        raise ReplyException("使用方式: /sticker del {sid}")
    StickerManager.get().delete(sid)
    await ctx.asend_reply_msg(f"已删除表情包 sid={sid}")


sticker_view = CmdHandler(["/sticker", "/表情包", "/stk"], logger)
sticker_view.check_cdrate(cd)


@sticker_view.handle()
async def _(ctx: HandlerContext):
    try:
        sid = int(ctx.get_args().strip())
    except Exception:
        raise ReplyException("使用方式: /sticker {sid}")
    s = StickerManager.get().find(sid)
    if s is None:
        raise ReplyException(f"表情包sid={sid}不存在")
    img_cq = await get_image_cq(s.path)
    lines = "\n".join([f"· {c['emotion']}，{c['scene']}" for c in s.caption]) if s.caption else "(无)"
    await ctx.asend_reply_msg(f"{img_cq}\n{lines}")


sticker_all = CmdHandler(["/all sticker", "/all stk", "/所有表情包"], logger)
sticker_all.check_cdrate(cd)


@sticker_all.handle()
async def _(ctx: HandlerContext):
    stickers = StickerManager.get().all()
    if not stickers:
        raise ReplyException("当前没有任何表情包")

    await run_in_pool(lambda: [s.ensure_thumb() for s in stickers])

    with Canvas(bg=FillBg(STICKER_THUMB_BG)).set_padding(8) as canvas:
        cols = max(1, int(math.sqrt(len(stickers) * 1.5)))
        with Grid(row_count=cols, hsep=6, vsep=6):
            for s in stickers:
                with VSplit().set_padding(0).set_sep(2).set_content_align("c").set_item_align("c"):
                    if s.thumb_path and os.path.exists(s.thumb_path):
                        ImageBox(s.thumb_path, size=STICKER_THUMB_SIZE, image_size_mode="fit").set_content_align("c")
                    else:
                        Spacer(w=STICKER_THUMB_SIZE[0], h=STICKER_THUMB_SIZE[1])
                    TextBox(f"{s.sid}", TextStyle(DEFAULT_FONT, 12, BLACK))

    await ctx.asend_reply_msg(await get_image_cq(await canvas.get_img(), low_quality=True))
