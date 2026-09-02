from typing import Union, Tuple, List, Optional, Dict, Any
from io import BytesIO
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageChops

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS
from PIL.ImageFont import ImageFont as Font
from dataclasses import dataclass, field, is_dataclass, fields
import os
import emoji.unicode_codes
import numpy as np
from copy import deepcopy
import math
from pilmoji import Pilmoji
from pilmoji import getsize as getsize_emoji
from pilmoji.source import GoogleEmojiSource, BaseSource
import emoji
from datetime import datetime, timedelta
import asyncio
from typing import get_type_hints
import colorsys
import random
import hashlib
import pickle
import glob
import io
import colour
import struct

from src.core import Config, get_cfg_or_value, get_logger
from .process_pool import ProcessPool, is_main_process
from .img_utils import adjust_image_alpha_inplace

global_config = Config("global")
logger = get_logger("draw")


def debug_print(*args, **kwargs):
    if global_config.get('painter.debug', False):
        print(*args, **kwargs, flush=True)

def get_memo_usage():
    if global_config.get('debug.painter', False):
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # 返回单位为MB
    return 0

def deterministic_hash(obj: Any, raise_error: bool = False) -> str:
    """
    计算复杂对象的确定性哈希值。
    
    Args:
        obj: 待哈希的对象
        raise_error: 遇到无法处理的类型时是否抛出异常。如果为 False，则哈希其类型名称。
    """
    hasher = hashlib.md5()
    # 用于检测循环引用：id(obj) -> recursion_depth
    seen = set()
    
    # 预编译 struct 格式，提高性能
    # 对应: bool(?), float(d), int-length(Q), array-shape(Q)
    STRUCT_BOOL = struct.Struct('?') 
    STRUCT_FLOAT = struct.Struct('>d') # Big-endian float
    STRUCT_Q = struct.Struct('>Q')     # Big-endian unsigned long long (8 bytes)

    def _update_bytes(b: bytes):
        hasher.update(b)

    def _update_str(s: str):
        b = s.encode('utf-8')
        # 写入长度前缀，防止边界模糊： "a", "b" -> 1a1b; "ab" -> 2ab
        _update_bytes(STRUCT_Q.pack(len(b)))
        _update_bytes(b)

    def _serialize(o: Any):
        # 1. 处理循环引用
        oid = id(o)
        if oid in seen:
            # 遇到循环引用，记录标记并返回
            _update_bytes(b'<RECURSION>')
            return
        
        # 只有容器和自定义对象才需要加入 seen，基本类型不需要（也不应该，因为 id 复用）
        is_container = isinstance(o, (dict, list, tuple, set, frozenset)) or hasattr(o, '__dict__') or hasattr(o, '__slots__')
        if is_container:
            seen.add(oid)

        try:
            # 2. 类型分发
            # 每个类型写入一个唯一的 Type Tag 前缀，防止类型混淆 (例如 "123" 和 123)
            
            if o is None:
                _update_bytes(b'N') # Null

            elif isinstance(o, bool):
                _update_bytes(b'B') # Bool
                _update_bytes(STRUCT_BOOL.pack(o))

            elif isinstance(o, int):
                _update_bytes(b'I')
                try:
                    # 尝试作为 8 字节长整型处理 (覆盖绝大多数业务场景)
                    _update_bytes(STRUCT_Q.pack(o) if o >= 0 else struct.pack('>q', o))
                except struct.error:
                    # 只有超大整数才走慢速通道
                    _update_bytes(hex(o).encode('ascii'))

            elif isinstance(o, float):
                _update_bytes(b'F') # Float
                # 使用 struct 保证二进制一致性 (注意：NaN != NaN 需要特殊处理)
                if np.isnan(o):
                    _update_bytes(b'nan')
                else:
                    _update_bytes(STRUCT_FLOAT.pack(o))

            elif isinstance(o, str):
                _update_bytes(b'S') # String
                _update_str(o)

            elif isinstance(o, (bytes, bytearray)):
                _update_bytes(b'Y') # Bytes
                _update_bytes(STRUCT_Q.pack(len(o)))
                _update_bytes(o)

            elif isinstance(o, (list, tuple)):
                _update_bytes(b'L' if isinstance(o, list) else b'T')
                _update_bytes(STRUCT_Q.pack(len(o)))
                for item in o:
                    _serialize(item)

            elif isinstance(o, dict):
                _update_bytes(b'D')
                _update_bytes(STRUCT_Q.pack(len(o)))
                # dict 无序，按键的可排序表示排一下，保证哈希确定
                kvs = []
                for k, v in o.items():
                    try:
                        k_repr = k if isinstance(k, (str, int, float, bool)) else str(k)
                        kvs.append((k_repr, k, v))
                    except Exception:
                        kvs.append((id(k), k, v))
                kvs.sort(key=lambda x: x[0])
                
                for _, k, v in kvs:
                    _serialize(k)
                    _serialize(v)

            elif isinstance(o, (set, frozenset)):
                _update_bytes(b'E') # Set
                _update_bytes(STRUCT_Q.pack(len(o)))
                # 集合是无序的，为了确定性，必须排序。
                # 无法直接对对象排序，先将所有元素序列化为 bytes 列表
                temp_hashes = []
                for item in o:
                    # 创建子 hasher 来获取每个元素的哈希
                    sub_hasher = hashlib.md5()
                    # 注意：这里需要传递 seen，防止集合内的循环引用
                    # 但为了简单，这里假设集合元素相对独立，或者在下一层处理
                    # 更好的方式是重构架构，这里简化处理：
                    try:
                        # 简单类型直接转 string 排序
                        if isinstance(item, (int, str, float, bool)):
                            temp_hashes.append((str(item).encode(), item))
                        else:
                            # 复杂类型，递归计算哈希值作为排序依据
                            sub_ret = deterministic_hash(item) # 可能会有性能损耗
                            temp_hashes.append((sub_ret.encode(), item))
                    except Exception:
                        pass # 忽略无法处理的项
                
                # 按 bytes 排序
                temp_hashes.sort(key=lambda x: x[0])
                for _, item in temp_hashes:
                    _serialize(item)

            elif isinstance(o, Image.Image):
                _update_bytes(b'P') # PIL
                # 包含尺寸、模式、数据
                _update_str(f"{o.size}:{o.mode}")
                # 使用 tobytes() 获取像素数据
                _update_bytes(STRUCT_Q.pack(len(o.tobytes())))
                _update_bytes(o.tobytes())

            elif hasattr(o, '__array__') and hasattr(o, 'dtype'):
                # Numpy Array
                _update_bytes(b'A') 
                # 必须包含 shape 和 dtype
                _update_str(str(o.shape))
                _update_str(str(o.dtype))
                
                # 如果数组太大，tobytes() 会很慢且耗内存。
                # 对于超大数组，建议只哈希其视图或采样，这里坚持确定性全量哈希
                # 使用 memoryview 避免复制
                if o.flags['C_CONTIGUOUS']:
                    _update_bytes(o.data)
                else:
                    _update_bytes(np.ascontiguousarray(o).data)

            elif is_dataclass(o) and not isinstance(o, type):
                _update_bytes(b'C') # Dataclass
                _update_str(o.__class__.__name__)
                for field in fields(o):
                    _serialize(field.name)
                    _serialize(getattr(o, field.name))

            else:
                # 通用对象处理
                _update_bytes(b'O') # Object
                _update_str(f"{o.__class__.__module__}.{o.__class__.__name__}")
                
                # 尝试获取状态
                data_to_hash = []
                
                if hasattr(o, '__dict__'):
                    data_to_hash += sorted([(k, v) for k, v in o.__dict__.items() if not k.startswith('_')])
                
                if hasattr(o, '__slots__'):
                    for slot in o.__slots__:
                        if not slot.startswith('_') and hasattr(o, slot):
                            data_to_hash.append((slot, getattr(o, slot)))
                
                # 如果没有状态，可能是可迭代对象
                if not data_to_hash and hasattr(o, '__iter__'):
                     _update_bytes(b'ITER')
                     for i in o:
                         _serialize(i)
                else:
                    # 序列化属性
                    # 排序属性名以保证顺序
                    data_to_hash.sort(key=lambda x: x[0])
                    for k, v in data_to_hash:
                        _serialize(k)
                        _serialize(v)
                    
                    if not data_to_hash:
                        if raise_error:
                            raise TypeError(f"Object of type {type(o)} is not hashable")
                        else:
                            _update_bytes(b'UNKNOWN')
                            
        finally:
            if is_container:
                seen.remove(oid)

    _serialize(obj)
    return hasher.hexdigest()


# =========================== 基础定义 =========================== #

PAINTER_CACHE_DIR = "data/utils/painter_cache/"
EMOJI_CACHE_DIR = "data/utils/emoji_cache/"

def _painter_process_num() -> int:
    return int(global_config.get("painter.process_num"))


PAINTER_PROCESS_NUM = _painter_process_num()


# ==================== Emoji 本地磁盘缓存 Source ==================== #

class CachedGoogleEmojiSource(BaseSource):
    """
    带本地磁盘缓存的 GoogleEmojiSource。
    优先读本地 EMOJI_CACHE_DIR/{hex}.png，不存在则从 Google CDN 拉取并缓存。
    线程安全（GIL 保护 dict，文件写入用临时文件原子替换）。
    """
    _cache_dir = EMOJI_CACHE_DIR
    _mem_cache: dict[str, bytes] = {}   # 进程内内存缓存，避免重复读盘
    _google_source = GoogleEmojiSource()

    def _cache_path(self, emoji: str) -> str:
        hex_name = "-".join(f"{ord(c):x}" for c in emoji)
        os.makedirs(self._cache_dir, exist_ok=True)
        return os.path.join(self._cache_dir, f"{hex_name}.png")

    def get_emoji(self, emoji: str) -> Optional[BytesIO]:
        # 1. 内存缓存
        if emoji in self._mem_cache:
            return BytesIO(self._mem_cache[emoji])

        # 2. 磁盘缓存
        path = self._cache_path(emoji)
        if os.path.exists(path):
            try:
                data = open(path, 'rb').read()
                self._mem_cache[emoji] = data
                return BytesIO(data)
            except Exception:
                pass

        # 3. 从 Google CDN 拉取
        try:
            result = self._google_source.get_emoji(emoji)
            if result is None:
                return None
            data = result.read()
        except Exception:
            return None

        # 4. 写入磁盘（原子写：先写临时文件再 rename）
        try:
            tmp_path = path + ".tmp"
            with open(tmp_path, 'wb') as f:
                f.write(data)
            os.replace(tmp_path, path)
        except Exception:
            pass

        self._mem_cache[emoji] = data
        return BytesIO(data)

    def get_discord_emoji(self, id: int) -> Optional[BytesIO]:
        return self._google_source.get_discord_emoji(id)


# 全局单例，整个进程共享内存缓存
_cached_emoji_source = CachedGoogleEmojiSource()

Color = Tuple[int, int, int, int]
Position = Tuple[int, int]
Size = Tuple[int, int]
LchColor = Tuple[float, float, float]

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
TRANSPARENT = (0, 0, 0, 0)
SHADOW = (0, 0, 0, 150)

ROUNDRECT_ANTIALIASING_TARGET_RADIUS_CFG = global_config.item('painter.roundrect_aa_target_radius')
EMOJI_SCALE_CFG = global_config.item('painter.emoji.scale')

FONT_DIR = "data/utils/fonts/"
DEFAULT_FONT = "msyh"
DEFAULT_BOLD_FONT = "SourceHanSansCN-Bold"
DEFAULT_HEAVY_FONT = "SourceHanSansCN-Heavy"
DEFAULT_EMOJI_FONT = "EmojiOneColor-SVGinOT"
_FONT_EXTS = ("", ".ttf", ".otf", ".ttc", ".otc")
_font_fallback_warned: set[str] = set()
_font_dir_logged = False


ALIGN_MAP = {
    'c': ('c', 'c'), 'l': ('l', 'c'), 'r': ('r', 'c'), 't': ('c', 't'), 'b': ('c', 'b'),
    'tl': ('l', 't'), 'tr': ('r', 't'), 'bl': ('l', 'b'), 'br': ('r', 'b'),
    'lt': ('l', 't'), 'lb': ('l', 'b'), 'rt': ('r', 't'), 'rb': ('r', 'b'), 
}


# =========================== 工具函数 =========================== #

@dataclass
class FontDesc:
    path: str
    size: int

@dataclass
class FontCacheEntry:
    font: Font
    last_used: datetime

FONT_CACHE_MAX_NUM_CFG = global_config.item('painter.font_cache_num')
font_cache: dict[str, FontCacheEntry] = {}
font_std_size_cache: dict[Font, Size] = {}

def crop_by_align(original_size, crop_size, align):
    w, h = original_size
    cw, ch = crop_size
    assert cw <= w and ch <= h, "Crop size must be smaller than original size"
    x, y = 0, 0
    xa, ya = ALIGN_MAP[align]
    if xa == 'l':
        x = 0
    elif xa == 'r':
        x = w - cw
    elif xa == 'c':
        x = (w - cw) // 2
    if ya == 't':
        y = 0
    elif ya == 'b':
        y = h - ch
    elif ya == 'c':
        y = (h - ch) // 2
    return x, y, x + cw, y + ch

def color_code_to_rgb(code: str) -> Color:
    if code.startswith("#"):
        code = code[1:]
    if len(code) == 3:
        return int(code[0], 16) * 16, int(code[1], 16) * 16, int(code[2], 16) * 16, 255
    elif len(code) == 6:
        return int(code[0:2], 16), int(code[2:4], 16), int(code[4:6], 16), 255
    raise ValueError("Invalid color code")

def rgb_to_color_code(rgb: Color) -> str:
    r, g, b = rgb[:3]
    return f"#{r:02x}{g:02x}{b:02x}"

def lerp_color(c1, c2, t):
    ret = []
    for i in range(len(c1)):
        ret.append(max(0, min(255, int(c1[i] * (1 - t) + c2[i] * t))))
    return tuple(ret)

def lerp_lch(c1: LchColor, c2: LchColor, t: float) -> LchColor:
    l = c1[0] * (1 - t) + c2[0] * t
    c = c1[1] * (1 - t) + c2[1] * t
    h1, h2 = c1[2], c2[2]
    if abs(h2 - h1) > 0.5:
        if h1 > h2:
            h2 += 1.0
        else:
            h1 += 1.0
    h = (h1 * (1 - t) + h2 * t) % 360.0
    return l, c, h

def adjust_color(c, r=None, g=None, b=None, a=None):
    c = list(c)
    if len(c) == 3: c.append(255)
    if r is not None: c[0] = r
    if g is not None: c[1] = g
    if b is not None: c[2] = b
    if a is not None: c[3] = a
    return tuple(c)

def get_font_desc(path: str, size: int) -> FontDesc:
    return FontDesc(path=path, size=size)

def _iter_font_dirs() -> list[str]:
    return [FONT_DIR]


def _font_request(name: str) -> tuple[str, bool]:
    stem, ext = os.path.splitext(name)
    if ext.lower() in {".ttf", ".otf", ".ttc", ".otc"}:
        name = stem
    want_bold = name.lower().endswith(("-bold", "-heavy"))
    if want_bold:
        name = name.rsplit("-", 1)[0]
    return name, want_bold


def _font_candidates(name: str) -> list[tuple[str, int]]:
    """(path, ttc_index)。粗体优先用独立 Bold 文件，否则走 ttc 的 1 号字面。"""
    base, want_bold = _font_request(name)
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    def add(path: str, index: int = 0):
        key = (path, index)
        if key not in seen:
            seen.add(key)
            out.append((path, index))

    add(name, 1 if want_bold else 0)
    stems = []
    if want_bold:
        stems.extend([f"{base}-Bold", f"{base}-Heavy", f"{base}bd", f"{base}b", base])
    stems.append(base)
    stems.append(name)
    for d in _iter_font_dirs():
        for stem in stems:
            for ext in _FONT_EXTS:
                path = os.path.join(d, stem + ext)
                lower = path.lower()
                if want_bold and lower.endswith((".ttc", ".otc")):
                    add(path, 1)
                    add(path, 0)
                else:
                    add(path, 0)
    return out


def _log_font_dir_once() -> None:
    global _font_dir_logged
    if _font_dir_logged:
        return
    _font_dir_logged = True
    missing = []
    for name in (DEFAULT_FONT, DEFAULT_BOLD_FONT, DEFAULT_HEAVY_FONT):
        if not any(os.path.exists(path) for path, _ in _font_candidates(name)):
            missing.append(name)
    if missing:
        logger.warning("中文字体缺失: %s，请放到 %s，否则中文会显示为方块", ", ".join(missing), FONT_DIR)


def _font_getsize(font: Font, text: str) -> tuple[int, int]:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _patch_font_getsize(font: Font) -> Font:
    if getattr(font, "_minori_getsize_patched", False):
        return font
    if callable(getattr(type(font), "getsize", None)):
        return font
    font.getsize = lambda text, *args, **kwargs: _font_getsize(font, text)  # type: ignore[method-assign]
    font._minori_getsize_patched = True  # type: ignore[attr-defined]
    return font


def get_font(path: str, size: int) -> Font:
    global font_cache
    _log_font_dir_once()
    key = f"{path}_{size}"
    name = path
    if key not in font_cache:
        font = None
        used_path = None
        for candidate, index in _font_candidates(name):
            if not os.path.exists(candidate):
                continue
            try:
                font = ImageFont.truetype(candidate, size, index=index)
                used_path = candidate if index == 0 else f"{candidate}#{index}"
                break
            except OSError:
                continue
        if font is None:
            fallbacks = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if ("Bold" in name or "Heavy" in name) else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            for fallback in fallbacks:
                if os.path.exists(fallback):
                    font = ImageFont.truetype(fallback, size)
                    used_path = fallback
                    break
        if font is None:
            raise FileNotFoundError(f"Font文件不存在: {name}（目录 {FONT_DIR}）")
        font = _patch_font_getsize(font)
        if used_path and FONT_DIR not in used_path.replace("\\", "/") and name not in _font_fallback_warned:
            _font_fallback_warned.add(name)
            if "dejavu" in used_path.lower():
                logger.warning("字体 %s 未找到，已回退 %s，中文会显示为方块。请把字体放到 %s", name, used_path, FONT_DIR)
            else:
                logger.info("字体 %s 从非默认目录加载: %s（建议放到 %s）", name, used_path, FONT_DIR)
        font_cache[key] = FontCacheEntry(
            font=font, 
            last_used=datetime.now(),
        )
        # 清理过期的字体缓存
        while len(font_cache) > FONT_CACHE_MAX_NUM_CFG.get():
            oldest_key = min(font_cache, key=lambda k: font_cache[k].last_used)
            removed = font_cache.pop(oldest_key)
            font_std_size_cache.pop(removed.font, None)
    return font_cache[key].font

def get_font_std_size(font: Font) -> Size:
    global font_std_size_cache
    if font not in font_std_size_cache:
        std_size = get_text_size(font, "哇")
        font_std_size_cache[font] = std_size
        return std_size
    return font_std_size_cache[font]

def has_emoji(text: str) -> bool:
    for c in text:
        if c in emoji.EMOJI_DATA:
            return True
    return False


def get_text_size(font: Font, text: str) -> Size:
    if not text: 
        return (0, 0)
    if has_emoji(text):
        return getsize_emoji(text, font=_patch_font_getsize(font), emoji_scale_factor=EMOJI_SCALE_CFG.get())
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
    
def get_text_width(font: Font, text: str) -> int:
    if not text:
        return 0
    if has_emoji(text):
        size = getsize_emoji(text, font=_patch_font_getsize(font), emoji_scale_factor=EMOJI_SCALE_CFG.get())
        return size[0]
    return font.getlength(text)

def get_text_offset(font: Font, text: str) -> Position:
    bbox = font.getbbox(text)
    return bbox[0], bbox[1]

def resize_keep_ratio(img: Image.Image, max_size: Union[int, float], mode='long', scale=None) -> Image.Image:
    """
    Resize image to keep the aspect ratio, with a maximum size.  
    mode in ['long', 'short', 'w', 'h', 'wxh', 'scale']
    """
    w, h = img.size
    if mode == 'long':
        if w > h:
            ratio = max_size / w
        else:
            ratio = max_size / h
    elif mode == 'short':
        if w > h:
            ratio = max_size / h
        else:
            ratio = max_size / w
    elif mode == 'w':
        ratio = max_size / w
    elif mode == 'h':
        ratio = max_size / h
    elif mode == 'wxh':
        ratio = math.sqrt(max_size / (w * h))
    elif mode == 'scale':
        ratio = max_size
    else:
        raise ValueError(f"Invalid mode: {mode}")
    if scale:
        ratio *= scale
    return img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.BILINEAR)

def resize_by_optional_size(img: Image.Image, size: Tuple[Optional[int], Optional[int]]) -> Image.Image:
    if size[0] is None and size[1] is None:
        return img
    if size[0] is None:
        if img.size[1] == size[1]:
            return img
        return resize_keep_ratio(img, size[1], mode='h')
    if size[1] is None:
        if img.size[0] == size[0]:
            return img
        return resize_keep_ratio(img, size[0], mode='w')
    if img.size[0] == size[0] and img.size[1] == size[1]:
        return img
    return img.resize(size, Image.Resampling.BILINEAR)

def srgb_to_oklch(rgb_colors: np.ndarray) -> np.ndarray:
    rgb_colors = rgb_colors.astype(np.float32) / 255.0
    srgb_linear = colour.sRGB_to_XYZ(rgb_colors)
    oklab_color = colour.XYZ_to_Oklab(srgb_linear)
    oklch_color = colour.Oklab_to_Oklch(oklab_color)
    return oklch_color

def oklch_to_srgb(lch_colors: np.ndarray) -> np.ndarray:
    oklch_color = lch_colors.astype(np.float32)
    oklab_color = colour.Oklch_to_Oklab(oklch_color)
    xyz_color = colour.Oklab_to_XYZ(oklab_color)
    rgb_color = colour.XYZ_to_sRGB(xyz_color)
    rgb_color = np.clip(rgb_color * 255.0, 0, 255).astype(np.uint8)
    return rgb_color


class Gradient:
    def _get_colors(self, size: Size) -> np.ndarray: 
        # [W, H, 4]
        raise NotImplementedError()

    def _lerp_color(self, t: np.ndarray, mode) -> np.ndarray:
        if mode in 'RGB_OR_RGBA':
            colors = (1 - t[:, :, np.newaxis]) * np.array(self.c1) + t[:, :, np.newaxis] * np.array(self.c2)
            return np.clip(colors, 0, 255).astype(np.uint8)
        elif mode in 'OKLCH':
            l = self.c1[0] * (1 - t) + self.c2[0] * t
            c = self.c1[1] * (1 - t) + self.c2[1] * t
            h1, h2 = self.c1[2] / 360.0, self.c2[2] / 360.0
            if abs(h2 - h1) > 0.5:
                if h1 > h2:
                    h2 += 1.0
                else:
                    h1 += 1.0
            h = (h1 * (1 - t) + h2 * t) % 1.0 * 360.0
            return np.stack((l, c, h), axis=-1)
        else:
            raise ValueError(f"Invalid Gradient color mode: {mode}")

    def get_img(self, size: Size, mask: Image.Image=None, mode='RGB_OR_RGBA') -> Image.Image:
        img = Image.fromarray(self._get_colors(size, mode), 'RGBA')
        if mask:
            assert mask.size == size, "Mask size must match image size"
            if mask.mode == 'RGBA':
                mask = mask.getchannel('A')
            else:
                mask = mask.convert('L')
            img.putalpha(mask)
        return img

    def get_array(self, size: Size, mode='RGB_OR_RGBA') -> np.ndarray:
        return self._get_colors(size, mode)

class LinearGradient(Gradient):
    def __init__(self, c1: Color, c2: Color, p1: Position, p2: Position, method: str = 'seperate'):
        self.c1 = c1
        self.c2 = c2
        self.p1 = p1
        self.p2 = p2
        self.method = method
        assert p1 != p2, "p1 and p2 cannot be the same point"

    def _get_colors(self, size: Size, mode: str) -> np.ndarray:
        w, h = size
        pixel_p1 = np.array((self.p1[1] * h, self.p1[0] * w))
        pixel_p2 = np.array((self.p2[1] * h, self.p2[0] * w))
        y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack((y_indices, x_indices), axis=-1) # (H, W, 2)
        if self.method == 'combine':
            gradient_vector = pixel_p2 - pixel_p1
            length_sq = np.sum(gradient_vector**2)
            vector_p1_to_pixel = coords - pixel_p1 # (H, W, 2)
            dot_product = np.sum(vector_p1_to_pixel * gradient_vector, axis=-1) # (H, W)
            t = dot_product / length_sq
        elif self.method == 'seperate': # seperate仅支持对角线/水平/垂直
            if abs(pixel_p1[0] - pixel_p2[0]) < 0.5:
                t = (coords[:, :, 1] - pixel_p1[1]) / (pixel_p2[1] - pixel_p1[1])
            elif abs(pixel_p1[1] - pixel_p2[1]) < 0.5:
                t = (coords[:, :, 0] - pixel_p1[0]) / (pixel_p2[0] - pixel_p1[0])
            else:
                vector_pixel_to_p1 = coords - pixel_p1
                vector_p2_to_p1 = pixel_p2 - pixel_p1
                t = np.average(vector_pixel_to_p1 / vector_p2_to_p1, axis=-1)
        else:
            raise ValueError(f"Invalid LinearGradient method: {self.method}")
        t_clamped = np.clip(t, 0, 1) 
        return self._lerp_color(t_clamped, mode)

class RadialGradient(Gradient):
    def __init__(self, c1: Color, c2: Color, center: Position, radius: float):
        self.c1 = c1
        self.c2 = c2
        self.center = center
        self.radius = radius

    def _get_colors(self, size: Size, mode: str) -> np.ndarray:
        w, h = size
        center = np.array(self.center) * np.array((w, h))
        y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack((x_indices, y_indices), axis=-1)
        dist = np.linalg.norm(coords - center, axis=-1) / self.radius
        dist = np.clip(dist, 0, 1)
        return self._lerp_color(dist, mode)
    

@dataclass
class AdaptiveTextColor:
    pixelwise: bool = False
    light: Color = WHITE
    dark: Color = BLACK
    threshold: float = 0.4

ADAPTIVE_WB = AdaptiveTextColor()
ADAPTIVE_SHADOW = AdaptiveTextColor(
    light=(255, 255, 255, 100), 
    dark=(0, 0, 0, 100), 
)


# =========================== 绘图类 =========================== #


SingleOrGradientLch = Union[LchColor, tuple[LchColor, LchColor]]

@dataclass
class RandomTriangleBgPreset:
    image_paths: list[str] = field(default_factory=list)
    image_weights: list[float] = field(default_factory=list)
    image_colors: list[Color] = field(default_factory=list)
    image_color_weights: list[float] = field(default_factory=list)
    scale: float = 1.0
    dense: float = 1.0
    alpha: float = 1.0
    time_colors: dict[int, SingleOrGradientLch] | None = None
    main_color: SingleOrGradientLch | None = None
    gradient_start: tuple[float, float] | None = (1.0, 0.0)
    gradient_end: tuple[float, float] | None = (0.0, 1.0)
    periods: list[tuple[str, str]] | None = None


@dataclass
class PainterOperation:
    offset: Position
    size: Size
    func: Union[str, callable]
    args: List
    exclude_on_hash: bool

    def image_to_id(self, img_dict: Dict[int, Image.Image]):
        if isinstance(self.args, tuple):
            self.args = list(self.args)
        for i in range(len(self.args)):
            if isinstance(self.args[i], Image.Image):
                img_id = id(self.args[i])
                img_dict[img_id] = self.args[i]
                self.args[i] = f"%%image%%{img_id}"
    
    def id_to_image(self, img_dict: Dict[int, Image.Image]):
        if isinstance(self.args, tuple):
            self.args = list(self.args)
        for i in range(len(self.args)):
            if isinstance(self.args[i], str) and self.args[i].startswith("%%image%%"):
                img_id = int(self.args[i][9:])
                self.args[i] = img_dict[img_id]


class Painter:
    
    def __init__(self, img: Image.Image = None, size: Tuple[int, int] = None):
        self.operations: List[PainterOperation] = []
        if img is not None:
            self.img = img
            self.size = img.size
        elif size is not None:
            self.img = None
            self.size = size
        else:
            raise ValueError("Either img or size must be provided")
        self.offset = (0, 0)
        self.w = self.size[0]
        self.h = self.size[1]
        self.region_stack = []


    def _text(
        self, 
        text: str, 
        pos: Position, 
        font: Font,
        fill: Color = BLACK,
        align: str = "left"
    ):
        std_size = get_font_std_size(font)
        if not has_emoji(text):
            draw = ImageDraw.Draw(self.img)
            text_offset = (0, -std_size[1])
            pos = (pos[0] - text_offset[0] + self.offset[0], pos[1] - text_offset[1] + self.offset[1])
            draw.text(pos, text, font=font, fill=fill, align=align, anchor='ls')
        else:
            with Pilmoji(self.img, source=_cached_emoji_source) as pilmoji:
                text_offset = (0, -std_size[1])
                offset = global_config.get('painter.emoji.offset')
                scale = global_config.get('painter.emoji.scale')
                offset = (int(offset[0] * std_size[1] / 32), int(offset[1] * std_size[1] / 32) - std_size[1])
                pos = (pos[0] - text_offset[0] + self.offset[0], pos[1] - text_offset[1] + self.offset[1])
                pilmoji.text(
                    pos, text, font=font, fill=fill, align=align, 
                    emoji_position_offset=offset, emoji_scale_factor=scale,
                    anchor='ls')
        return self
    
    def _get_aa_roundrect(
        self,
        size: Size, 
        fill: Color,
        radius: int, 
        stroke: Color=None, 
        stroke_width: int=1,
        corners = (True, True, True, True), 
        margin: int | tuple[int, int, int, int] = 0,    # left, top, right, bottom
    ) -> Image.Image:
        width, height = size
        if isinstance(margin, int):
            margin = (margin, margin, margin, margin)
        ml, mt, mr, mb = margin

        width, height = width - 1, height - 1
        radius = min(radius, width // 2, height // 2)
        realsize = (width + ml + mr + 1, height + mt + mb + 1)

        def getbox(x1, y1, x2, y2):
            return (x1 + ml, y1 + mt, x2 + ml, y2 + mt)
        def getpos(x, y):
            return (x + ml, y + mt)

        # 特殊情况：半径为0，直接绘制矩形
        if radius <= 0:
            img = Image.new('RGBA', realsize, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle(getbox(0, 0, width, height), fill=fill, outline=stroke, width=stroke_width)
            return img

        img = Image.new('RGBA', realsize, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 绘制中心矩形区域
        if fill:
            draw.rectangle(getbox(radius, 0, width - radius, height), fill=fill)
            draw.rectangle(getbox(0, radius, width, height - radius), fill=fill)

        # 绘制四条直边的描边
        if stroke and stroke_width > 0:
            draw.rectangle(getbox(radius, 0, width - radius, stroke_width), fill=stroke) # 上
            draw.rectangle(getbox(radius, height - stroke_width, width - radius, height), fill=stroke) # 下
            draw.rectangle(getbox(0, radius, stroke_width, height - radius), fill=stroke) # 左
            draw.rectangle(getbox(width - stroke_width, radius, width, height - radius), fill=stroke) # 右

        # 抗锯齿的缩放比例
        aa_scale = max(1, math.ceil(ROUNDRECT_ANTIALIASING_TARGET_RADIUS_CFG.get() / radius))
        aa_radius = radius * aa_scale
        aa_stroke_width = stroke_width * aa_scale

        # 创建一个放大的、带抗锯齿的圆角模板 (左上角)，并缩小回原始大小
        corner_aa = None
        if any(corners):
            corner_canvas = Image.new('RGBA', (aa_radius * 2, aa_radius * 2), (0, 0, 0, 0))
            corner_draw = ImageDraw.Draw(corner_canvas)
            corner_draw.rounded_rectangle(
                (0, 0, aa_radius * 2, aa_radius * 2),
                radius=aa_radius,
                fill=fill,
                outline=stroke,
                width=aa_stroke_width,
                corners=(True, True, True, True)
            )
            corner_canvas = corner_canvas.crop((0, 0, aa_radius, aa_radius))
            corner_aa = corner_canvas.resize((radius + 1, radius + 1), Image.Resampling.BICUBIC)
        
        # 创建一个普通的直角模板 (不需要圆角的角落)
        sharp_corner = None
        if not all(corners):
            sharp_corner = Image.new('RGBA', (radius + 1, radius + 1), (0, 0, 0, 0))
            sharp_draw = ImageDraw.Draw(sharp_corner)
            if fill:
                sharp_draw.rectangle((0, 0, radius + 1, radius + 1), fill=fill)
            if stroke and stroke_width > 0:
                sharp_draw.rectangle((0, 0, radius + 1, stroke_width), fill=stroke) # 上
                sharp_draw.rectangle((0, 0, stroke_width, radius + 1), fill=stroke) # 左

        tl, tr, br, bl = corners
        corner = corner_aa if tl else sharp_corner
        img.paste(corner, getpos(0, 0))
        corner = (corner_aa if tr else sharp_corner).transpose(Image.FLIP_LEFT_RIGHT)
        img.paste(corner, getpos(width - radius, 0))
        corner = (corner_aa if br else sharp_corner).transpose(Image.ROTATE_180)
        img.paste(corner, getpos(width - radius, height - radius))
        corner = (corner_aa if bl else sharp_corner).transpose(Image.FLIP_TOP_BOTTOM)
        img.paste(corner, getpos(0, height - radius))
        return img


    @staticmethod
    def _execute(operations: List[PainterOperation], img: Image.Image, size: Tuple[int, int], image_dict: Dict[str, Image.Image]) -> Image.Image:
        start_time = datetime.now()
        debug_print(f"Sub process enter memory usage: {get_memo_usage()} MB")
        if img is None:
            img = Image.new('RGBA', size, TRANSPARENT)
        p = Painter(img, size)
        for op in operations:
            op.id_to_image(image_dict)
            # debug_print(f"Executing: {op}")
            p.offset = op.offset
            p.size = op.size
            p.w, p.h = op.size
            func = getattr(p, op.func) if isinstance(op.func, str) else op.func
            kwargs = {}
            for key, value in get_type_hints(func).items():
                if value == Painter:
                    kwargs[key] = p
            t = datetime.now()
            func(*op.args, **kwargs)
            if global_config.get('painter.log_operation', False):
                debug_print(f"Method {op.func} executed, mem: {get_memo_usage()} MB, time: {datetime.now() - t}")
        debug_print(f"Sub process use time: {datetime.now() - start_time}")
        return p.img

    async def get(self, cache_key: str=None) -> Image.Image:
        # 使用缓存
        if cache_key is not None:
            t = datetime.now()
            debug_print(f"Cache key: {cache_key}")
            op_hash = await asyncio.to_thread(deterministic_hash, {"key": cache_key, "op": self.operations})
            debug_print(f"Cache key: {cache_key}, op_hash: {op_hash}, elapsed: {datetime.now() - t}")

            paths = glob.glob(os.path.join(PAINTER_CACHE_DIR, f"{cache_key}__*.png"))
            if paths:
                path = paths[0]
                if path.endswith(f"{cache_key}__{op_hash}.png"):
                    # 如果hash相同则直接返回缓存的图片
                    debug_print(f"Using cached image: {path}")
                    img = Image.open(path)
                    img.load()
                    return img
                else:
                    # 否则清空缓存并重新绘图
                    for p in paths:
                        try: 
                            os.remove(p)
                        except Exception as e: 
                            print(f"Failed to remove cache file {p}: {e}")
                    debug_print(f"Cache mismatch, removed {len(paths)} files")

        debug_print(f"Main process memory usage: {get_memo_usage()} MB")

        # 收集所有图片对象到字典中
        image_dict = {}
        for op in self.operations:
            op.image_to_id(image_dict)
        total_img_size = 0
        for img in image_dict.values():
            total_img_size += img.size[0] * img.size[1] * 4
        debug_print(f"image_dict len: {len(image_dict)}, total size: {total_img_size//1024//1024} MB")

        # 执行绘图操作
        t = datetime.now()

        pool = await _ensure_painter_pool()
        if pool is not None:
            self.img = await pool.submit(Painter._execute, self.operations, self.img, self.size, image_dict)
        else:
            self.img = await asyncio.to_thread(Painter._execute, self.operations, self.img, self.size, image_dict)

        self.operations = []
        debug_print(f"Painter executed in {datetime.now() - t}")

        # 保存缓存
        if cache_key is not None:
            try:
                cache_path = os.path.join(PAINTER_CACHE_DIR, f"{cache_key}__{op_hash}.png")
                os.makedirs(PAINTER_CACHE_DIR, exist_ok=True)
                self.img.save(cache_path, format='PNG')
            except:
                debug_print(f"Failed to save cache for {cache_key}")

        return self.img
    
    def add_operation(self, func: Union[str, callable], exclude_on_hash: bool, args: List[Any]):
        self.operations.append(PainterOperation(
            offset=self.offset,
            size=self.size,
            func=func,
            args=list(args),
            exclude_on_hash=exclude_on_hash,
        ))
        return self

    @staticmethod
    def clear_cache(cache_key: str) -> int:
        paths = glob.glob(os.path.join(PAINTER_CACHE_DIR, f"{cache_key}__*.png"))
        ok = 0
        for p in paths:
            try: 
                os.remove(p)
                ok += 1
            except Exception as e: 
                print(f"Failed to remove cache file {p}: {e}")
        return ok
    
    @staticmethod
    def get_cache_key_mtimes() -> Dict[str, datetime]:
        paths = glob.glob(os.path.join(PAINTER_CACHE_DIR, f"*.png"))
        cache_keys = {}
        for p in paths:
            mtime = os.path.getmtime(p)
            cache_key = os.path.basename(p).split('__')[0]
            cache_keys[cache_key] = datetime.fromtimestamp(mtime)
        return cache_keys


    def set_region(self, pos: Position, size: Size):
        assert isinstance(pos[0], int) and isinstance(pos[1], int), "Position must be integer"
        assert isinstance(size[0], int) and isinstance(size[1], int), "Size must be integer"
        self.region_stack.append((self.offset, self.size))
        self.offset = pos
        self.size = size
        self.w = size[0]
        self.h = size[1]
        return self

    def shrink_region(self, dlt: Position):
        pos = (self.offset[0] + dlt[0], self.offset[1] + dlt[1])
        size = (self.size[0] - dlt[0] * 2, self.size[1] - dlt[1] * 2)
        return self.set_region(pos, size)

    def expand_region(self, dlt: Position):
        pos = (self.offset[0] - dlt[0], self.offset[1] - dlt[1])
        size = (self.size[0] + dlt[0] * 2, self.size[1] + dlt[1] * 2)
        return self.set_region(pos, size)

    def move_region(self, dlt: Position, size: Size = None):
        offset = (self.offset[0] + dlt[0], self.offset[1] + dlt[1])
        size = size or self.size
        return self.set_region(offset, size)

    def restore_region(self, depth=1):
        if not self.region_stack:
            self.offset = (0, 0)
            self.size = self.img.size
            self.w = self.img.size[0]
            self.h = self.img.size[1]
        else:
            self.offset, self.size = self.region_stack.pop()
            self.w = self.size[0]
            self.h = self.size[1]
        if depth > 1:
            return self.restore_region(depth - 1)
        return self


    def text(
        self, 
        text: str, 
        pos: Position, 
        font: Union[FontDesc, Font],
        fill: Union[Color, LinearGradient, AdaptiveTextColor] = BLACK,
        align: str = "left",
        exclude_on_hash: bool = False,
    ):
        """
        绘制文本

        Parameters:
            text: 要绘制的单行文本内容
            pos: 文本位置 (x, y)
            font: 字体，可以是FontDesc或PIL ImageFont对象
            fill: 填充颜色，可以是Color/LinearGradient/AdaptiveTextColor
            align: 对齐方式，'left', 'center', 'right'
            exclude_on_hash: 是否在哈希计算中排除此操作
        """
        return self.add_operation("_impl_text", exclude_on_hash, (text, pos, font, fill, align))
        
    def paste(
        self, 
        sub_img: Image.Image,
        pos: Position, 
        size: Size = None,
        use_shadow: bool = False,
        shadow_width: int = 8,
        shadow_alpha: float = 0.6,
        exclude_on_hash: bool = False,
    ) -> Image.Image:
        """
        直接粘贴图像

        Parameters:
            sub_img: 要粘贴的子图像
            pos: 粘贴位置 (x, y)
            size: 调整子图像大小 (width, height)
            use_shadow: 是否使用阴影效果
            shadow_width: 阴影宽度
            shadow_alpha: 阴影透明度
            exclude_on_hash: 是否在哈希计算中排除此操作
        """
        return self.add_operation("_impl_paste", exclude_on_hash, (sub_img, pos, size, use_shadow, shadow_width, shadow_alpha))

    def paste_with_alphablend(
        self, 
        sub_img: Image.Image,
        pos: Position, 
        size: Size = None,
        alpha: float = None,
        use_shadow: bool = False,
        shadow_width: int = 8,
        shadow_alpha: float = 0.6,
        exclude_on_hash: bool = False,
    ) -> Image.Image:
        """
        以Alpha混合的方式粘贴图像

        Parameters:
            sub_img: 要粘贴的子图像
            pos: 粘贴位置 (x, y)
            size: 调整子图像大小 (width, height)
            alpha: 透明度，范围0.0-1.0
            use_shadow: 是否使用阴影效果
            shadow_width: 阴影宽度
            shadow_alpha: 阴影透明度
            exclude_on_hash: 是否在哈希计算中排除此操作
        """
        return self.add_operation("_impl_paste_with_alphablend", exclude_on_hash, (sub_img, pos, size, alpha, use_shadow, shadow_width, shadow_alpha))

    def rect(
        self, 
        pos: Position, 
        size: Size, 
        fill: Union[Color, Gradient], 
        stroke: Color=None, 
        stroke_width: int=1,
        exclude_on_hash: bool = False
    ):
        """
        绘制矩形

        Parameters:
            pos: 矩形位置 (x, y)
            size: 矩形大小 (width, height)
            fill: 填充颜色，可以是Color或Gradient
            stroke: 描边颜色
            stroke_width: 描边宽度
            exclude_on_hash: 是否在哈希计算中排除此操作
        """
        return self.add_operation("_impl_rect", exclude_on_hash, (pos, size, fill, stroke, stroke_width))
        
    def roundrect(
        self, 
        pos: Position, 
        size: Size, 
        fill: Union[Color, Gradient],
        radius: int, 
        stroke: Color=None, 
        stroke_width: int=1,
        corners = (True, True, True, True),
        exclude_on_hash: bool = False
    ):
        """
        绘制圆角矩形

        Parameters:
            pos: 矩形位置 (x, y)
            size: 矩形大小 (width, height)
            fill: 填充颜色，可以是Color或Gradient
            radius: 圆角半径
            stroke: 描边颜色
            stroke_width: 描边宽度
            corners: 四个角的圆角启用状态，顺序为左上、右上、右下、左下
            exclude_on_hash: 是否在哈希计算中排除此操作
        """
        return self.add_operation("_impl_roundrect", exclude_on_hash, (pos, size, fill, radius, stroke, stroke_width, corners))

    def pieslice(
        self,
        pos: Position,
        size: Size,
        start_angle: float,
        end_angle: float,
        fill: Color,
        stroke: Color=None,
        stroke_width: int=1,
        exclude_on_hash: bool = False
    ):
        """
        绘制扇形

        Parameters:
            pos: 扇形位置 (x, y)
            size: 扇形大小 (width, height)
            start_angle: 起始角度（度数制）
            end_angle: 结束角度（度数制）
            fill: 填充颜色
            stroke: 描边颜色
            stroke_width: 描边宽度
            exclude_on_hash: 是否在哈希计算中排除此操作
        """
        return self.add_operation("_impl_pieslice", exclude_on_hash, (pos, size, start_angle, end_angle, fill, stroke, stroke_width))

    def blurglass_roundrect(
        self, 
        pos: Position, 
        size: Size, 
        fill: Color,
        radius: int, 
        blur: float=4,
        shadow_width: int=6,
        shadow_alpha: float=0.3,
        corners = (True, True, True, True),
        exclude_on_hash: bool = False
    ):
        """
        绘制模糊玻璃圆角矩形

        Parameters:
            pos: 矩形位置 (x, y)
            size: 矩形大小 (width, height)
            fill: 填充颜色
            radius: 圆角半径
            blur: 模糊半径
            shadow_width: 阴影宽度
            shadow_alpha: 阴影透明度
            corners: 四个角的圆角启用状态，顺序为左上、右上、右下、左下
            exclude_on_hash: 是否在哈希计算中排除此操作
        """
        return self.add_operation("_impl_blurglass_roundrect", exclude_on_hash, (pos, size, fill, radius, blur, shadow_width, shadow_alpha, corners))

    def draw_random_triangle_bg(
        self, 
        preset_config_name: str, 
        size_fixed_rate: float = 0.0,
        dt: datetime | None = None,
        exclude_on_hash: bool = False,
    ):
        """
        绘制随机三角形背景

        Parameters:
            preset_config_name: 预设配置名称
            size_fixed_rate: 随机三角形固定大小比率，0.0代表大小随画布大小变化，1.0表示总是使用相同大小
            dt: 用于时间相关颜色计算的时间点，设置为None则使用当前时间
            exclude_on_hash: 是否在哈希计算中排除此操作
        """
        return self.add_operation("_impl_draw_random_triangle_bg", exclude_on_hash, (preset_config_name, size_fixed_rate, dt))


    def _impl_text(
        self, 
        text: str, 
        pos: Position, 
        font: Union[FontDesc, Font],
        fill: Union[Color, LinearGradient, AdaptiveTextColor] = BLACK,
        align: str = "left"
    ):
        def adjust_overlay_alpha_by_color(overlay: Image.Image, color: Color):
            if len(color) < 4 or color[3] == 255:
                return
            overlay_alpha = overlay.getchannel('A')
            overlay_alpha = Image.eval(overlay_alpha, lambda a: int(a * color[3] / 255))
            overlay.putalpha(overlay_alpha)

        if isinstance(font, FontDesc):
            font = get_font(font.path, font.size)

        if isinstance(fill, LinearGradient):
            gradient = fill
            adaptive = None
            fill = BLACK
        elif isinstance(fill, AdaptiveTextColor):
            gradient = None
            adaptive = fill
            fill = fill.light[:3]
        else:
            gradient = None
            adaptive = None

        if (len(fill) == 3 or fill[3] == 255) and not gradient and not adaptive:
            # 不透明，非渐变，非高对比度颜色
            self._text(text, pos, font, fill, align)
        else:
            text_size = get_text_size(font, text)
            overlay_size = (text_size[0] + 10, text_size[1] + 10)
            overlay = Image.new('RGBA', overlay_size, (0, 0, 0, 0))
            p = Painter(overlay)
            p._text(text, (0, 0), font, fill=fill, align=align)

            if gradient:
                # 渐变颜色
                gradient_img = gradient.get_img(overlay_size, overlay)
                overlay = gradient_img

            elif adaptive:
                # 自适应颜色
                dark_overlay = Image.new('RGBA', overlay_size, (0, 0, 0, 0))
                dark_p = Painter(dark_overlay)
                dark_p._text(text, (0, 0), font, fill=adaptive.dark[:3], align=align)

                adjust_overlay_alpha_by_color(overlay, adaptive.light)
                adjust_overlay_alpha_by_color(dark_overlay, adaptive.dark)

                bg_img = self.img.crop((
                    pos[0] + self.offset[0], 
                    pos[1] + self.offset[1], 
                    pos[0] + self.offset[0] + overlay_size[0], 
                    pos[1] + self.offset[1] + overlay_size[1]
                ))

                if adaptive.pixelwise:
                    gray = bg_img.filter(ImageFilter.BoxBlur(radius=8)).convert('L')
                else:
                    avg_color = np.array(bg_img).reshape(-1, 4).mean(axis=0)
                    gray = Image.new('RGB', bg_img.size, tuple(avg_color[:3].astype(int))).convert('L')

                threshold = int(adaptive.threshold * 255)
                mask = gray.point(lambda p: 255 if p > threshold else 0, 'L')
                overlay.paste(dark_overlay, (0, 0), mask)

            elif fill[3] < 255:
                # 半透明颜色
                adjust_overlay_alpha_by_color(overlay, fill)

            self.img.alpha_composite(overlay, (pos[0] + self.offset[0], pos[1] + self.offset[1]))

        return self
        
    def _impl_paste(
        self, 
        sub_img: Image.Image,
        pos: Position, 
        size: Size = None,
        use_shadow: bool = False,
        shadow_width: int = 6,
        shadow_alpha: float = 0.6,
    ) -> Image.Image:
        if size and size != sub_img.size:
            sub_img = sub_img.resize(size)
        if sub_img.mode not in ('RGB', 'RGBA'):
            sub_img = sub_img.convert('RGBA')

        if use_shadow:
            w, h = sub_img.size
            sw = shadow_width
            lw, lh = w + sw * 2, h + sw * 2
            # 获取和图像相同形状的阴影mask
            shadow_mask = Image.new('L', (lw, lh), 0)
            shadow_mask.paste(Image.new('L', sub_img.size, int(255 * shadow_alpha)), (sw, sw), sub_img)
            # 模糊获取阴影
            blurred_shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=sw // 2))
            # 删除内部阴影
            inner_mask = ImageChops.invert(shadow_mask)
            blurred_shadow_mask = ImageChops.multiply(blurred_shadow_mask, inner_mask)
            # 贴入原图
            shadow = Image.new('RGBA', (lw, lh), (0, 0, 0, 255))
            shadow.putalpha(blurred_shadow_mask)
            self.img.alpha_composite(shadow, (pos[0] + self.offset[0] - sw, pos[1] + self.offset[1] - sw))

        if sub_img.mode == 'RGBA':
            self.img.paste(sub_img, (pos[0] + self.offset[0], pos[1] + self.offset[1]), sub_img)
        else:
            self.img.paste(sub_img, (pos[0] + self.offset[0], pos[1] + self.offset[1]))
        return self

    def _impl_paste_with_alphablend(
        self, 
        sub_img: Image.Image,
        pos: Position, 
        size: Size = None,
        alpha: float = None,
        use_shadow: bool = False,
        shadow_width: int = 6,
        shadow_alpha: float = 0.6,
    ) -> Image.Image:
        if size and size != sub_img.size:
            sub_img = sub_img.resize(size)
        pos = (pos[0] + self.offset[0], pos[1] + self.offset[1])
        overlay = Image.new('RGBA', sub_img.size, (0, 0, 0, 0))
        overlay.paste(sub_img, (0, 0))
        if alpha is not None:
            overlay_alpha = overlay.getchannel('A')
            overlay_alpha = Image.eval(overlay_alpha, lambda a: int(a * alpha))
            overlay.putalpha(overlay_alpha)

        if use_shadow:
            w, h = overlay.size
            sw = shadow_width
            lw, lh = w + sw * 2, h + sw * 2
            # 获取和图像相同形状的阴影mask
            shadow_mask = Image.new('L', (lw, lh), 0)
            shadow_mask.paste(Image.new('L', overlay.size, int(255 * shadow_alpha)), (sw, sw), overlay)
            # 模糊获取阴影
            blurred_shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=sw // 2))
            # 删除内部阴影
            inner_mask = ImageChops.invert(shadow_mask)
            blurred_shadow_mask = ImageChops.multiply(blurred_shadow_mask, inner_mask)
            # 贴入原图
            shadow = Image.new('RGBA', (lw, lh), (0, 0, 0, 255))
            shadow.putalpha(blurred_shadow_mask)
            self.img.alpha_composite(shadow, (pos[0] - sw, pos[1] - sw))

        self.img.alpha_composite(overlay, pos)
        return self

    def _impl_rect(
        self, 
        pos: Position, 
        size: Size, 
        fill: Union[Color, Gradient], 
        stroke: Color=None, 
        stroke_width: int=1,
    ):
        if min(size) <= 0:
            return self

        if isinstance(fill, Gradient):
            gradient = fill
            fill = BLACK
        else:
            gradient = None
        
        pos = (pos[0] + self.offset[0], pos[1] + self.offset[1])
        bbox = pos + (pos[0] + size[0], pos[1] + size[1])

        if fill[3] == 255 and not gradient:
            draw = ImageDraw.Draw(self.img)
            draw.rectangle(bbox, fill=fill, outline=stroke, width=stroke_width)
        else:
            overlay_size = (size[0] + 1, size[1] + 1)
            overlay = Image.new('RGBA', overlay_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle((0, 0, size[0], size[1]), fill=fill, outline=stroke, width=stroke_width)
            if gradient:
                gradient_img = gradient.get_img(overlay_size, overlay)
                overlay = gradient_img
            self.img.alpha_composite(overlay, (pos[0], pos[1]))

        return self
        
    def _impl_roundrect(
        self, 
        pos: Position, 
        size: Size, 
        fill: Union[Color, Gradient],
        radius: int, 
        stroke: Color=None, 
        stroke_width: int=1,
        corners = (True, True, True, True),
    ):
        if min(size) <= 0:
            return self

        if isinstance(fill, Gradient):
            gradient = fill
            fill = BLACK
        else:
            gradient = None

        pos = (pos[0] + self.offset[0], pos[1] + self.offset[1])

        overlay = self._get_aa_roundrect(size, fill, radius, stroke, stroke_width, corners)

        if gradient:
            gradient_img = gradient.get_img(overlay.size, overlay)
            overlay = gradient_img

        self.img.alpha_composite(overlay, (pos[0], pos[1]))
        
        return self

    def _impl_pieslice(
        self,
        pos: Position,
        size: Size,
        start_angle: float,
        end_angle: float,
        fill: Color,
        stroke: Color=None,
        stroke_width: int=1,
    ):
        if min(size) <= 0 or start_angle >= end_angle:
            return self

        if isinstance(fill, Gradient):
            gradient = fill
            fill = BLACK
        else:
            gradient = None

        pos = (pos[0] + self.offset[0], pos[1] + self.offset[1])
        bbox = pos + (pos[0] + size[0], pos[1] + size[1])

        if fill[3] == 255 and not gradient:
            draw = ImageDraw.Draw(self.img)
            draw.pieslice(bbox, start_angle, end_angle, fill=fill, width=stroke_width, outline=stroke)
        else:
            overlay_size = (size[0] + 1, size[1] + 1)
            overlay = Image.new('RGBA', overlay_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.pieslice((0, 0, size[0], size[1]), start_angle, end_angle, fill=fill, width=stroke_width, outline=stroke)
            if gradient:
                gradient_img = gradient.get_img(overlay_size, overlay)
                overlay = gradient_img
            self.img.alpha_composite(overlay, (pos[0], pos[1]))
        
        return self

    def _impl_blurglass_roundrect(
        self, 
        pos: Position, 
        size: Size, 
        fill: Color,
        radius: int, 
        blur: float=4,
        shadow_width: int=6,
        shadow_alpha: float=0.3,
        corners = (True, True, True, True),
        edge_strength: float=0.6,
    ):  
        if min(size) <= 0:
            return self

        sw = shadow_width
        pos = (pos[0] + self.offset[0], pos[1] + self.offset[1])
        draw_pos = (pos[0] - sw, pos[1] - sw)
        draw_size = (size[0] + sw * 2, size[1] + sw * 2)

        alpha = fill[3] if isinstance(fill, tuple) and len(fill) == 4 else 0
        bg_offset = int(24 * min(blur / 6, alpha / 200))
        bg_offset = min(bg_offset, draw_size[0] - bg_offset, draw_size[1] - bg_offset)
        bg_region = (
            pos[0] + bg_offset // 2,
            pos[1] + bg_offset // 2,
            pos[0] + size[0] - bg_offset // 2,
            pos[1] + size[1] - bg_offset // 2,
        )
        
        if isinstance(fill, Gradient):
            # 填充渐变色
            bg = fill.get_img((bg_region[2] - bg_region[0], bg_region[3] - bg_region[1]))
        elif len(fill) == 3 or fill[3] == 255:
            # 填充纯色
            if len(fill) == 3: fill = (*fill, 255)
            bg = Image.new('RGBA', (bg_region[2] - bg_region[0], bg_region[3] - bg_region[1]), fill)
        else:
            # 复制pos位置的size大小的原图模糊并混合颜色
            bg = self.img.crop(bg_region)
            if blur > 0:
                # 适当缩小背景
                downsample = max(1, blur // 2)
                if downsample > 1:
                    bg = bg.resize(
                        (bg.width // downsample, bg.height // downsample),
                        Image.Resampling.BILINEAR
                    )
                blur_method = ImageFilter.GaussianBlur if downsample >= 2 else ImageFilter.BoxBlur
                bg = bg.filter(blur_method(radius=blur / downsample))
            bg.alpha_composite(Image.new('RGBA', bg.size, tuple(fill)))

        # 超分绘制圆角矩形
        overlay = self._get_aa_roundrect(
            size=size,
            fill=BLACK,
            radius=radius,
            corners=corners,
            margin=sw,
        )

        # 取得mask
        inner_mask = overlay.copy()
        bg_mask = overlay.crop((sw, sw, sw + size[0], sw + size[1]))

        # 通过模糊底图获取阴影，然后删除内部阴影
        adjust_image_alpha_inplace(overlay, shadow_alpha, method='multiply')
        # 只模糊四个边以提升性能
        swb = int(sw * 1.5)
        overlay_left = overlay.crop((0, 0, min(swb, overlay.width), overlay.height))
        overlay_right = overlay.crop((overlay.width - min(swb, overlay.width), 0, overlay.width, overlay.height))
        overlay_upper = overlay.crop((0, 0, overlay.width, min(swb, overlay.height)))
        overlay_lower = overlay.crop((0, overlay.height - min(swb, overlay.height), overlay.width, overlay.height))
        overlay_left = overlay_left.filter(ImageFilter.GaussianBlur(radius=sw * 0.5))
        overlay_right = overlay_right.filter(ImageFilter.GaussianBlur(radius=sw * 0.5))
        overlay_upper = overlay_upper.filter(ImageFilter.GaussianBlur(radius=sw * 0.5))
        overlay_lower = overlay_lower.filter(ImageFilter.GaussianBlur(radius=sw * 0.5))
        overlay.paste(overlay_left, (0, 0))
        overlay.paste(overlay_right, (overlay.width - overlay_right.width, 0))
        overlay.paste(overlay_upper, (0, 0))
        overlay.paste(overlay_lower, (0, overlay.height - overlay_lower.height))
        overlay = ImageChops.multiply(overlay, ImageChops.invert(inner_mask))

        # 用圆角矩形mask裁剪并粘贴背景
        bg = bg.resize(size, Image.Resampling.BILINEAR)
        bg.putalpha(bg_mask.getchannel('A')) 
        overlay.alpha_composite(bg, (sw, sw))

        # 边缘效果
        if edge_strength is not None and edge_strength > 0:
            edge_width = min(4, min(draw_size) // 16, radius // 2)
            if edge_width > 0:
                # 绘制超分圆角矩形边缘底图
                ew = edge_width
                edge_overlay = self._get_aa_roundrect(
                    size=size,
                    fill=None,
                    radius=radius,
                    stroke=WHITE,
                    stroke_width=ew,
                    corners=corners,
                    margin=sw,
                )

                # 生成各个位置的渐变色块矩形（左上角，左边，上边，右下角，右边，下边）通过坐标换算保证渐变颜色过渡正确
                alpha1, alpha2 = int(255 * edge_strength), int(255 * edge_strength * 0.75)
                lt_points, rb_points = ((0, 0), (0.8, 0.4)), ((0.6, 0.8), (1.0, 1.0))
                lt_colors = ((255, 255, 255, alpha1), (255, 255, 255, 0))
                rb_colors = ((255, 255, 255, 0), (255, 255, 255, alpha2))
                w, h = draw_size[0], draw_size[1]
                def get_grad_p(p1, p2, pos, size):
                    p1, p2 = (p1[0] * w, p1[1] * h), (p2[0] * w, p2[1] * h)
                    newp1 = ((p1[0] - pos[0]) / size[0], (p1[1] - pos[1]) / size[1])
                    newp2 = ((p2[0] - pos[0]) / size[0], (p2[1] - pos[1]) / size[1])
                    return { 'p1': newp1, 'p2': newp2 }
                
                edge_color_overlay = Image.new('RGBA', draw_size, TRANSPARENT)
                t_pos, t_size = (sw, sw), (w - sw * 2, ew)
                edge_color_t = LinearGradient(*lt_colors, **get_grad_p(*lt_points, t_pos, t_size)).get_img(t_size)
                edge_color_overlay.paste(edge_color_t, t_pos)
                l_pos, l_size = (sw, sw), (ew, h - sw * 2)
                edge_color_l = LinearGradient(*lt_colors, **get_grad_p(*lt_points, l_pos, l_size)).get_img(l_size)
                edge_color_overlay.paste(edge_color_l, l_pos)
                lt_pos, lt_size = (sw, sw), (radius, radius)
                edge_color_lt = LinearGradient(*lt_colors, **get_grad_p(*lt_points, lt_pos, lt_size)).get_img(lt_size)
                edge_color_overlay.paste(edge_color_lt, lt_pos)

                r_pos, r_size = (w - ew - sw, sw), (ew, h - sw * 2)
                edge_color_r = LinearGradient(*rb_colors, **get_grad_p(*rb_points, r_pos, r_size)).get_img(r_size)
                edge_color_overlay.paste(edge_color_r, r_pos)
                b_pos, b_size = (sw, h - ew - sw), (w - sw * 2, ew)
                edge_color_b = LinearGradient(*rb_colors, **get_grad_p(*rb_points, b_pos, b_size)).get_img(b_size)
                edge_color_overlay.paste(edge_color_b, b_pos)
                rb_pos, rb_size = (w - radius - sw, h - radius - sw), (radius, radius)
                edge_color_rb = LinearGradient(*rb_colors, **get_grad_p(*rb_points, rb_pos, rb_size)).get_img(rb_size)
                edge_color_overlay.paste(edge_color_rb, rb_pos)

                # 渐变色和边缘底图相乘
                edge_overlay = ImageChops.multiply(edge_overlay, edge_color_overlay)
                overlay.alpha_composite(edge_overlay)

        # 贴回原图
        self.img.alpha_composite(overlay, (draw_pos[0], draw_pos[1]))
        return self

    def _impl_draw_random_triangle_bg(self, preset_config_name: str, size_fixed_rate: float, dt: datetime | None):
        def get_timecolor(timecolors: dict[int, SingleOrGradientLch], t: datetime) -> SingleOrGradientLch:
            """
            从时间颜色列表中获取当前时间的插值颜色(lch)
            """
            tcs = [(hour, c) for hour, c in timecolors.items()]
            tcs.sort(key=lambda x: x[0])
            if t.hour < tcs[0][0]:
                return tcs[0][1]
            elif t.hour >= tcs[-1][0]:
                return tcs[-1][1]
            for i in range(0, len(tcs) - 1):
                if t.hour >= tcs[i][0] and t.hour < tcs[i + 1][0]:
                    hour1, c1 = tcs[i]
                    hour2, c2 = tcs[i + 1]
                    t1 = datetime(t.year, t.month, t.day, hour1)
                    if hour2 == 24: t2 = datetime(t.year, t.month, t.day, 0) + timedelta(days=1)
                    else:           t2 = datetime(t.year, t.month, t.day, hour2)
                    if len(c1) == 3: c1 = (c1, c1)
                    if len(c2) == 3: c2 = (c2, c2)
                    x = (t - t1) / (t2 - t1)
                    return lerp_lch(c1[0], c2[0], x), lerp_lch(c1[1], c2[1], x)
                
        # 选择预设
        now = dt or datetime.now()
        preset_config = Config(preset_config_name)
        preset = None
        for i, p in enumerate(preset_config.get('presets', []), 1):
            p = RandomTriangleBgPreset(**p)
            assert p.main_color or p.time_colors, f"No main_color or time_colors defined in preset #{i} in {preset_config.path}"
            if not p.periods:
                preset = p
                break
            for start_s, end_s in p.periods:
                start = datetime.strptime(start_s, "%m-%d %H:%M").replace(year=now.year)
                end = datetime.strptime(end_s, "%m-%d %H:%M").replace(year=now.year)
                if start <= now <= end:
                    preset = p
                    break
            if preset:
                break
        assert preset, f"No valid preset found in {preset_config.path}"

        # 加载预设图片
        images, image_weights = [], []
        for i, image_path in enumerate(preset.image_paths):
            try:
                img = Image.open(image_path).convert("RGBA")
                images.append(img)
                image_weights.append(preset.image_weights[i])
            except:
                print(f"Warning: failed to load random triangle bg image: {image_path}")
        
        # 确定主颜色
        if preset.time_colors:
            main_color = get_timecolor(preset.time_colors, now)
        else:
            main_color = preset.main_color
        if len(main_color) == 3:
            l1, c1, h1 = main_color
            l2, c2, h2 = main_color
        else:
            l1, c1, h1 = main_color[0]
            l2, c2, h2 = main_color[1]
        
        # 渐变背景
        w, h = self.size
        scale = max(1, min(w, h) // 64)
        bg = LinearGradient(
            c1=(l1, c1, h1),
            c2=(l2, c2, h2),
            p1=preset.gradient_start, p2=preset.gradient_end,
            method='seperate',
        ).get_array((w // scale, h // scale), mode='OKLCH')
        bg = oklch_to_srgb(bg)
        bg = Image.fromarray(bg, 'RGB').convert('RGBA')
        bg = bg.resize((w, h), Image.Resampling.LANCZOS)

        def draw_tri(x, y, rot, size, alpha):
            if not images: return
            img = random.choices(images, weights=image_weights, k=1)[0]
            color = random.choices(preset.image_colors, weights=preset.image_color_weights, k=1)[0]
            color = (*color, alpha) if len(color) == 3 else (*color[:3], color[3] * alpha // 255)
            img = img.resize((size, size), Image.Resampling.BILINEAR)
            img = img.rotate(rot, expand=True)
            img = ImageChops.multiply(img, Image.new("RGBA", img.size, color))
            bg.alpha_composite(img, (int(x) - img.width // 2, int(y) - img.height // 2))

        factor = min(w, h) / 2048 * 1.5
        size_factor = (1.0 + (factor - 1.0) * (1.0 - size_fixed_rate)) * preset.scale
        dense_factor = (1.0 + (factor * factor - 1.0) * size_fixed_rate) * preset.dense

        def rand_tri(num, sz):
            # 生成网格随机分布
            positions: list[tuple[int, int]] = []
            nx = int(math.sqrt(num))
            ny = int(num / nx) + 1
            dx, dy = w / nx, h / ny
            for ix in range(nx):
                for iy in range(ny):
                    px = ix / nx * w + dx * 0.5 + random.uniform(-dx * 0.4, dx * 0.4)
                    py = iy / ny * h + dy * 0.5 + random.uniform(-dy * 0.4, dy * 0.4)
                    positions.append((px, py))

            for i in range(num):
                x, y = positions[i]
                if x < 0 or x >= w or y < 0 or y >= h:
                    continue
                rot = random.uniform(0, 360)
                size = max(1, min(1000, int(random.normalvariate(sz[0], sz[1]))))
                dist = (((x - w // 2) / w * 2) ** 2 + ((y - h // 2) / h * 2) ** 2)
                size = int(size * dist)

                alpha = random.normalvariate(50, 200) 

                # 大小影响透明度
                size_alpha_factor, std_size_lower, std_size_upper = 1.0, 64 * size_factor, 128 * size_factor
                if size < std_size_lower:
                    size_alpha_factor = size / std_size_lower
                if size > std_size_upper:
                    size_alpha_factor = 1.0 - (size - std_size_upper * 1.5) / (std_size_upper * 1.5)
                alpha *= max(0, min(1.2, size_alpha_factor))    

                # 背景亮度影响透明度
                lightness_alpha_factor = max(((l1 + l2) * 0.5), 0.3)
                alpha *= lightness_alpha_factor   

                # 随机一些特别亮的三角形
                if random.random() < 0.05 and size > std_size_lower:
                    alpha = 255 * lightness_alpha_factor

                # 预设透明度最终调整
                alpha = alpha * preset.alpha

                alpha = int(alpha)
                if alpha <= 10:
                    continue

                draw_tri(x, y, rot, size, alpha)

        rand_tri(int(20 * dense_factor), (128 * size_factor, 16 * size_factor))
        rand_tri(int(100 * dense_factor), (100 * size_factor, 16 * size_factor))

        self.img.paste(bg, self.offset)


_painter_pool = None
_painter_pool_lock: asyncio.Lock | None = None
_painter_pool_started = False


def _pool_lock() -> asyncio.Lock:
    global _painter_pool_lock
    if _painter_pool_lock is None:
        _painter_pool_lock = asyncio.Lock()
    return _painter_pool_lock


async def _ensure_painter_pool():
    """首次真正出图时再拉起绘图进程池。"""
    global _painter_pool, _painter_pool_started
    if _painter_pool_started:
        return _painter_pool
    async with _pool_lock():
        if _painter_pool_started:
            return _painter_pool
        _painter_pool_started = True
        n = _painter_process_num()
        if n > 0 and is_main_process():
            try:
                pool = ProcessPool(n, name="MinoriBot-Drawing")
                _painter_pool = pool
                if pool.backend == "process":
                    logger.info("MinoriBot-Drawing initialized successfully")
                else:
                    err = pool.fallback_error
                    logger.warning(
                        "MinoriBot‑Drawing process startup failed, fallback to thread pool: %s",
                        type(err).__name__ if err else "unknown",
                    )
            except Exception:
                _painter_pool = None
                logger.print_exc("MinoriBot-Drawing initialized failed")
        else:
            logger.info("MinoriBot-Drawing initialized successfully")
        return _painter_pool

