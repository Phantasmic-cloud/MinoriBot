from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import glob
import math
import os
import random
import shutil
import subprocess
import zipfile

import numpy as np
from PIL import Image

from src.utils import *


config = Config("gallery")
logger = get_logger("gallery")
file_db = get_file_db("data/gallery/gallery.json")
cd = ColdDown(file_db, logger)
gbl = get_group_black_list(file_db, logger, "gallery")

THUMBNAIL_SIZE = (64, 64)
SIZE_LIMIT_MB_CFG = config.item('size_limit_mb')
HASH1_DIFFERENCE_THRESHOLD_CFG = config.item('hash1_difference_threshold')
HASH2_DIFFERENCE_THRESHOLD_CFG = config.item('hash2_difference_threshold')
GALLERY_PICS_DIR = 'data/gallery/{name}/'
PIC_EXTS = ['.jpg', '.jpeg', '.png', '.gif']
ADD_LOG_FILE = 'data/gallery/add.log'
THUMBNAIL_BG_COLOR = (230, 240, 255, 255)


add_history_db = get_file_db("data/gallery/add_history.json")


# ======================= 逻辑处理 ======================= # 

class GalleryMode(Enum):
    """画廊模式：编辑 / 查看 / 关闭。"""
    Edit = 'edit'
    View = 'view'
    Off = 'off'


@dataclass
class GalleryPic:
    """画廊里的一张图。"""
    gall_name: str
    pid: int
    path: str
    hash1: str = None
    hash2: str = None
    thumb_path: str | None = None

    @classmethod
    def load(cls, data: dict) -> 'GalleryPic':
        return cls(
            gall_name=data['gall_name'],
            pid=data['pid'],
            path=data['path'],
            hash1=data.get('hash1', None),
            hash2=data.get('hash2', None),
            thumb_path=data.get('thumb_path', None),
        )

    def calc_hash(self):
        """算感知哈希，用来查重。"""
        image = Image.open(self.path)
        # 如果存在A通道：alphablend到纯白色背景上
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            image = image.convert('RGBA').resize((64, 64), Image.Resampling.BILINEAR)
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            bg.alpha_composite(image)
            image = bg
        image = image.convert('RGB')
        image = image.resize((16, 16), Image.Resampling.BILINEAR).convert('L')
        # hash2: 用于MAE计算，直接转为b64字符串
        self.hash2 = image.tobytes().hex()
        # hash1: 用于快速比较，计算64位感知哈希
        image = image.resize((8, 8), Image.Resampling.BILINEAR)
        pixels = np.array(image).flatten()
        avg = pixels.mean()
        bits = 0
        for idx, p in enumerate(pixels):
            if p >= avg:
                bits |= 1 << (63 - idx)
        self.hash1 = f"{bits:016x}"

    def is_same(self, other: 'GalleryPic') -> bool:
        """两张图是否判为重复。"""
        # 通过hash1快速排除不同的图片
        if (int(self.hash1, 16) ^ int(other.hash1, 16)).bit_count() > HASH1_DIFFERENCE_THRESHOLD_CFG.get():
            return False
        # hash2精确判断
        img1 = np.frombuffer(bytes.fromhex(self.hash2), dtype=np.uint8)
        img2 = np.frombuffer(bytes.fromhex(other.hash2), dtype=np.uint8)
        diff = np.sum(np.abs(img1.astype(np.int16) - img2.astype(np.int16)))
        return diff <= HASH2_DIFFERENCE_THRESHOLD_CFG.get()

    def ensure_thumb(self):
        """没有缩略图就现生成一张。"""
        try:
            if self.thumb_path is None:
                name = os.path.basename(self.path)
                self.thumb_path = os.path.join(os.path.dirname(self.path), f"{name}_thumb.jpg")
            if not os.path.exists(self.thumb_path):
                img = Image.open(self.path).convert('RGBA')
                img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                thumb = Image.new('RGBA', img.size, THUMBNAIL_BG_COLOR)
                thumb.alpha_composite(img)
                thumb.convert('RGB').save(self.thumb_path, format='JPEG', optimize=True, quality=85)
        except Exception as e:
            logger.warning(f'生成画廊图片 {self.pid} 缩略图失败: {e}')
            self.thumb_path = None


class GalleryPicRepeatedException(Exception):
    """上传时撞上已有相似图。"""
    def __init__(self, pid: int):
        super().__init__(f'画廊中已存在相似图片(pid={pid})')
        self.pid = pid


@dataclass
class Gallery:
    """一个画廊：名字、别名、模式和图片列表。"""
    name: str
    aliases: list[str]
    mode: GalleryMode
    pics_dir: str
    cover_pid: int | None = None
    pics: list[GalleryPic] = field(default_factory=list)


class GalleryManager:
    """画廊仓库：加载、增删改查、查重。"""
    _mgr: 'GalleryManager' = None

    def __init__(self):
        self.pid_top = 0
        self.galleries: dict[str, Gallery] = {}

    def _load(self):
        self.pid_top = file_db.get('pid_top', 0)
        self.galleries = {}
        for name, g in file_db.get('galleries', {}).items():
            self.galleries[name] = Gallery(
                name=g['name'],
                aliases=g.get('aliases', []),
                cover_pid=g.get('cover_pid', None),
                mode=GalleryMode(g.get('mode', 'edit')),
                pics_dir=g['pics_dir'],
                pics=[GalleryPic.load(p) for p in g.get('pics', [])],
            )
        logger.info(f'成功加载{len(self.galleries)}个画廊, pid_top={self.pid_top}')

    def _save(self):
        file_db.set('pid_top', self.pid_top)
        file_db.set('galleries', { name: asdict(g) for name, g in self.galleries.items() })
        file_db.save()

    def _check_name(self, name: str) -> bool:
        if not name or len(name) > 32:
            return False
        if any(c in name for c in r'\/:*?"<>| '):
            return False
        if name.isdigit():
            return False
        return True

    async def _async_check_duplicated(self, pic: GalleryPic, gallery: Gallery) -> int | None:
        def check():
            for p in gallery.pics:
                if pic.is_same(p):
                    return p.pid
            return None
        return await run_in_pool(check)
        

    @classmethod
    def get(cls) -> 'GalleryManager':
        if cls._mgr is None:
            cls._mgr = GalleryManager()
            cls._mgr._load()
        return cls._mgr


    def get_all_galls(self) -> dict[str, Gallery]:
        """
        获取所有画廊
        """
        return self.galleries

    def find_gall(self, name_or_alias: str, raise_if_nofound: bool = False) -> Gallery | None:
        """
        通过名称或别名查找画廊
        """
        for g in self.galleries.values():
            if g.name == name_or_alias or name_or_alias in g.aliases:
                return g
        if raise_if_nofound:
            if not name_or_alias:
                raise ReplyException('画廊名称不能为空')
            raise ReplyException(f'画廊\"{name_or_alias}\"不存在')
        return None

    def open_gall(self, name: str):
        """
        创建一个新画廊
        """
        assert self._check_name(name), f'画廊名称\"{name}\"无效'
        assert self.find_gall(name) is None, f'画廊\"{name}\"已存在'
        gall = Gallery(
            name=name,
            aliases=[],
            mode=GalleryMode.Edit,
            pics_dir=GALLERY_PICS_DIR.format(name=name),
            pics=[],
        )
        self.galleries[name] = gall
        self._save()

    def close_gall(self, name_or_alias: str):
        """
        删除一个画廊
        """
        g = self.find_gall(name_or_alias)
        assert g is not None, f'画廊\"{name_or_alias}\"不存在'
        self.galleries.pop(g.name)
        self._save()

    def add_gall_alias(self, name_or_alias: str, alias: str):
        """
        为画廊添加一个别名
        """
        assert self._check_name(alias), f'别名\"{alias}\"无效'
        g = self.find_gall(name_or_alias)
        assert g is not None, f'画廊\"{name_or_alias}\"不存在'
        assert self.find_gall(alias) is None, f'别名\"{alias}\"已被占用'
        g.aliases.append(alias)
        self._save()

    def del_gall_alias(self, name_or_alias: str, alias: str):
        """
        删除画廊的一个别名
        """
        g = self.find_gall(name_or_alias)
        assert g is not None, f'画廊\"{name_or_alias}\"不存在'
        assert alias in g.aliases, f'别名\"{alias}\"不存在'
        g.aliases.remove(alias)
        self._save()

    def change_gall_mode(self, name_or_alias: str, mode: GalleryMode) -> tuple[GalleryMode, GalleryMode]:
        """
        修改画廊的模式，返回(旧模式, 新模式)
        """
        g = self.find_gall(name_or_alias)
        assert g is not None, f'画廊\"{name_or_alias}\"不存在'
        old_mode = g.mode
        g.mode = mode
        self._save()
        return old_mode, g.mode

    def find_pic(self, pid: int, raise_if_nofound: bool = False) -> GalleryPic | None:
        """
        通过图片ID查找图片
        """
        if pid < 0:
            pids = []
            for g in self.galleries.values():
                for p in g.pics:
                    pids.append(p.pid)
            pids.sort()
            if pid < -len(pids):
                if raise_if_nofound:
                    raise ReplyException(f'画廊仅有{len(pids)}张图片')
                return None
            pid = pids[pid]
        for g in self.galleries.values():
            for p in g.pics:
                if p.pid == pid:
                    return p
        if raise_if_nofound:
            raise ReplyException(f'画廊图片pid={pid}不存在')
        return None

    async def async_add_pic(self, name_or_alias: str, img_path: str, check_duplicated: bool = True) -> int:
        """
        向画廊添加一张图片，将会直接拷贝img_path的图片，返回图片ID
        """
        g = self.find_gall(name_or_alias)
        assert g is not None, f'画廊\"{name_or_alias}\"不存在'

        pic = GalleryPic(
            gall_name=g.name, 
            pid=self.pid_top+1,
            path=img_path, 
        )
        await run_in_pool(pic.calc_hash)
    
        if check_duplicated:
            if sim_pid := await self._async_check_duplicated(pic, g):
                raise GalleryPicRepeatedException(sim_pid)

        self.pid_top += 1
        _, ext = os.path.splitext(os.path.basename(img_path))
        time_str = datetime.now().strftime('%Y-%m-%d_%H-%M%-S')
        dst_path = create_parent_folder(os.path.join(g.pics_dir, f"{time_str}_{self.pid_top}{ext}"))
        shutil.copy2(img_path, dst_path)

        pic.path = dst_path
        g.pics.append(pic)
        await run_in_pool(pic.ensure_thumb)
        self._save()
        return self.pid_top

    async def async_replace_pic(self, pid: int, img_path: str, check_duplicated: bool = True) -> int:
        """
        替换画廊中的一张图片，返回图片ID
        """
        p = self.find_pic(pid)
        assert p is not None, f'图片ID {pid} 不存在'
        g = self.find_gall(p.gall_name)
        assert g is not None, f'图片ID {pid} 所属画廊\"{p.gall_name}\"不存在'

        new_pic = GalleryPic(
            gall_name=g.name, 
            pid=p.pid, 
            path=img_path, 
        )
        await run_in_pool(new_pic.calc_hash)

        if check_duplicated:
            if sim_pid := await self._async_check_duplicated(new_pic, g):
                if sim_pid != pid:
                    raise GalleryPicRepeatedException(sim_pid)

        # 删除旧文件
        try:
            if os.path.exists(p.path):
                os.remove(p.path)
            if p.thumb_path and os.path.exists(p.thumb_path):
                os.remove(p.thumb_path)
        except Exception as e:
            logger.warning(f'删除画廊图片 {pid} 文件失败: {get_exc_desc(e)}')

        # 复制新文件
        _, ext = os.path.splitext(os.path.basename(img_path))
        time_str = datetime.now().strftime('%Y-%m-%d_%H-%M%-S')
        dst_path = create_parent_folder(os.path.join(g.pics_dir, f"{time_str}_{p.pid}{ext}"))
        shutil.copy2(img_path, dst_path)

        # 更新信息
        p.path = dst_path
        p.hash1 = new_pic.hash1
        p.hash2 = new_pic.hash2
        p.thumb_path = None
        await run_in_pool(p.ensure_thumb)

        self._save()
        return p.pid

    def del_pic(self, pid: int) -> int:
        """
        从画廊删除一张图片，返回被删除的图片ID
        """
        p = self.find_pic(pid)
        assert p is not None, f'图片ID {pid} 不存在'
        g = self.find_gall(p.gall_name)
        g.pics.remove(p)
        try:
            if os.path.exists(p.path):
                os.remove(p.path)
            if p.thumb_path and os.path.exists(p.thumb_path):
                os.remove(p.thumb_path)
        except Exception as e:
            logger.warning(f'删除画廊图片 {pid} 文件失败: {get_exc_desc(e)}')
        self._save()
        return p.pid

    async def async_reload_gall(self, name_or_alias: str) -> tuple[list[int], list[int]]:
        """
        从画廊的图片目录重新加载图片（不去重），返回[新加载的图片pids, 失效的图片pids]
        """
        g = self.find_gall(name_or_alias)
        assert g is not None, f'画廊\"{name_or_alias}\"不存在'
        new_pids, del_pids = [], []
        # 检查新增的图片
        for file in glob.glob(os.path.join(g.pics_dir, '*')):
            if '_thumb' in file:
                continue
            for p in g.pics:
                if os.path.abspath(p.path) == os.path.abspath(file):
                    continue
            _, ext = os.path.splitext(os.path.basename(file))
            if ext.lower() in PIC_EXTS:
                self.pid_top += 1
                pic = GalleryPic(
                    gall_name=g.name, 
                    pid=self.pid_top, 
                    path=file,
                )
                await run_in_pool(pic.calc_hash)
                g.pics.append(pic)
                new_pids.append(pic.pid)

        # 检查失效的图片
        for pic in g.pics[:]:
            if not os.path.exists(pic.path):
                g.pics.remove(pic)
                del_pids.append(pic.pid)
        self._save()
        return new_pids, del_pids
    
    def set_cover_pic(self, name_or_alias: str, pid: int):
        """
        设置画廊封面图片
        """
        g = self.find_gall(name_or_alias)
        assert g is not None, f'画廊\"{name_or_alias}\"不存在'
        p = self.find_pic(pid)
        assert p is not None and p.gall_name == g.name, f'图片pid={pid}不属于画廊\"{g.name}\"'
        g.cover_pid = pid
        self._save()

    async def async_check_gallery(self, name_or_alias: str, rehash: bool) -> dict[int, list[int]]:
        """
        重新检查画廊重复图片，返回一个字典，key为首个图片id，value为重复图片id列表
        """
        def check():
            g = self.find_gall(name_or_alias)
            assert g is not None, f'画廊\"{name_or_alias}\"不存在'
            
            ret: dict[int, tuple[GalleryPic, list[int]]] = {}  # ret[pid] = (first_pic, dup_pids)
            for pic in g.pics[:]:
                if rehash:
                    pic.calc_hash()

                sim_pid = None
                for k, v in ret.items():
                    if pic.is_same(v[0]):
                        sim_pid = k
                        break

                if sim_pid is not None:
                    ret[sim_pid][1].append(pic.pid)
                else:
                    ret[pic.pid] = (pic, [])

            if rehash:
                self._save()
            ret = { k : v[1] for k, v in ret.items() if v[1] }
            return ret
        return await run_in_pool(check)


# 处理本地文件用于添加到画廊
def process_image_for_gallery(path: str, sub_type: int):
    img = open_image(path)
    # 如果是表情并且是静态图，可能获取到jpg，需要手动转换为静态gif
    need_to_gif = (sub_type and not is_animated(img))

    # 根据限制进行缩放
    scaled = False
    size_limit = SIZE_LIMIT_MB_CFG.get()
    filesize_mb = os.path.getsize(path) / (1024 * 1024)
    if filesize_mb > size_limit:
        pixels = get_image_pixels(img)
        img = limit_image_by_pixels(img, int(pixels * size_limit / filesize_mb))
        scaled = True

    if need_to_gif:
        # 转换为静态gif
        save_transparent_static_gif(img, path)
    elif scaled:
        if is_animated(img):
            save_transparent_gif(img, get_gif_duration(img), path)
        else:
            img.save(path)
        new_size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"缩放过大的图片 {filesize_mb:.2f}M -> {new_size_mb:.2f}M")

# 添加上传记录，返回记录id
def add_user_add_history(user_id: int, pids: list[int]) -> int:
    history = add_history_db.get('history', [])
    record = {
        'id': len(history) + 1,
        'uid': user_id,
        'pids': pids,
        'ts': datetime.now().timestamp(),
        'reverted': False,
    }
    history.append(record)
    add_history_db.set('history', history)
    return record['id']

# 根据记录id撤销上传，返回(记录, 成功pids, 失败pids)
def revert_user_add_history(hid: int) -> tuple[dict, list[int], list[int]]:
    history = add_history_db.get('history', [])
    h = find_by(history, 'id', hid)
    assert_and_reply(h is not None, f'上传#{hid}不存在')
    assert_and_reply(not h['reverted'], f'上传#{hid}已被撤销')

    ok_list, err_list = [], []
    for pid in h['pids']:
        try:
            GalleryManager.get().del_pic(pid)
            ok_list.append(pid)
        except Exception as e:
            logger.warning(f'撤销上传记录#{hid}时删除图片pid={pid}失败: {get_exc_desc(e)}')
            err_list.append(pid)

    h['reverted'] = True
    add_history_db.set('history', history)
    return h, ok_list, err_list

# 撤销某个用户的最近一次上传，返回(记录, 成功pids, 失败pids)
def revert_user_last_add_history(user_id: int) -> tuple[dict, list[int], list[int]]:
    history = add_history_db.get('history', [])
    user_histories = [h for h in reversed(history) if h['uid'] == user_id and not h['reverted']]
    assert_and_reply(user_histories, '你没有可撤销的上传记录')
    h = user_histories[0]

    expired_hours = config.get('user_recent_revert_expired_hours')
    time = datetime.fromtimestamp(h['ts'])
    assert_and_reply((datetime.now() - time) < timedelta(hours=expired_hours), f'最近一次上传记录已超过{expired_hours}小时，无法撤销')

    ok_list, err_list = [], []
    for pid in h['pids']:
        try:
            GalleryManager.get().del_pic(pid)
            ok_list.append(pid)
        except Exception as e:
            logger.warning(f'撤销上传记录#{h["id"]}时删除图片pid={pid}失败: {get_exc_desc(e)}')
            err_list.append(pid)

    h['reverted'] = True
    add_history_db.set('history', history)
    return h, ok_list, err_list

# 根据记录id获取上传记录
def get_user_add_history(hid: int) -> dict:
    history = add_history_db.get('history', [])
    h = find_by(history, 'id', hid)
    assert_and_reply(h is not None, f'上传#{hid}不存在')
    return h


# ======================= 指令处理 ======================= # 

gall_open = CmdHandler([
    '/gall open',
], logger)
gall_open.check_cdrate(cd).check_wblist(gbl).check_superuser()
@gall_open.handle()
async def _(ctx: HandlerContext):
    name = ctx.get_args().strip()
    GalleryManager.get().open_gall(name)
    await ctx.asend_reply_msg(f'画廊\"{name}\"创建成功')


gall_close = CmdHandler([
    '/gall close',
], logger)
gall_close.check_cdrate(cd).check_wblist(gbl).check_superuser()
@gall_close.handle()
async def _(ctx: HandlerContext):
    name = ctx.get_args().strip()
    GalleryManager.get().close_gall(name)
    await ctx.asend_reply_msg(f'画廊\"{name}\"删除成功')


gall_mode = CmdHandler([
    '/gall mode',
], logger)
gall_mode.check_cdrate(cd).check_wblist(gbl).check_superuser()
@gall_mode.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip().split()
    if len(args) == 1:
        mode = GalleryManager.get().find_gall(args[0], raise_if_nofound=True).mode.value
        return await ctx.asend_reply_msg(f'画廊\"{args[0]}\"当前模式: {mode}')
    if len(args) > 2:
        raise ReplyException('使用方式: /gall mode 画廊名称 模式(edit/view/off)')
    name, mode = args
    old, new = GalleryManager.get().change_gall_mode(name, GalleryMode(mode))
    await ctx.asend_reply_msg(f'画廊\"{name}\"模式修改成功: {old.value} -> {new.value}')


gall_del = CmdHandler([
    '/gall del', '/gall remove',
], logger)
gall_del.check_cdrate(cd).check_wblist(gbl).check_superuser()
@gall_del.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()

    try:
        if '-' in args:
            l, r = args.split('-', 1)
            l, r = int(l), int(r)
            pids = list(range(l, r+1))
        else:
            pids = [int(s) for s in args.split()]
            l, r = None, None
        assert pids
    except:
        raise ReplyException("""
使用方式: 
/gall del 123 456 -1 -2 ...
/gall del 123-456 (最多连续20张)
""".strip())
    
    # 安全限制
    if l is not None:
        assert_and_reply(r - l < 20, '一次最多删除20张连续图片')
    gall_names = set()
    for pid in pids:
        if pic := GalleryManager.get().find_pic(pid, raise_if_nofound=False):
            gall_names.add(pic.gall_name)
    if len(gall_names) > 1:
        raise ReplyException('禁止跨画廊删除图片')
    
    ok_list, err_list = [], []
    for pid in pids:
        try:
            pid = GalleryManager.get().del_pic(pid)
            ok_list.append(pid)
        except Exception as e:
            logger.warning(f'删除画廊图片pid={pid}失败: {get_exc_desc(e)}')
            err_list.append(pid)

    msg = ""
    if ok_list:
        msg += f'{len(ok_list)}张图片删除成功:\n'
        msg += ' '.join([str(pid) for pid in ok_list]) + '\n'
    if err_list:
        msg += f'{len(err_list)}张图片删除失败:\n'
        msg += ' '.join([str(pid) for pid in err_list]) + '\n'
    await ctx.asend_fold_msg_adaptive(msg.strip())


gall_reload = CmdHandler([
    '/gall reload', '/gall update',
], logger)
gall_reload.check_cdrate(cd).check_wblist(gbl).check_superuser()
@gall_reload.handle()
async def _(ctx: HandlerContext):
    name = ctx.get_args().strip()
    new_pids, del_pids = await GalleryManager.get().async_reload_gall(name)
    msg = f'画廊\"{name}\"重新加载完成\n新增图片: {len(new_pids)}张\n失效图片: {len(del_pids)}张'
    await ctx.asend_reply_msg(msg)


gall_alias_add = CmdHandler([
    '/gall alias add', '/gall add alias',
], logger, priority=1)
gall_alias_add.check_cdrate(cd).check_wblist(gbl).check_superuser()
@gall_alias_add.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip().split()
    if len(args) != 2:
        raise ReplyException('使用方式: /gall alias add 画廊名称 别名')
    name, alias = args
    GalleryManager.get().add_gall_alias(name, alias)
    await ctx.asend_reply_msg(f'画廊\"{name}\"添加别名\"{alias}\"成功')


gall_alias_del = CmdHandler([
    '/gall alias del', '/gall alias remove', '/gall del alias', '/gall remove alias',
], logger)
gall_alias_del.check_cdrate(cd).check_wblist(gbl).check_superuser()
@gall_alias_del.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip().split()
    if len(args) != 2:
        raise ReplyException('使用方式: /gall alias del 画廊名称 别名')
    name, alias = args
    GalleryManager.get().del_gall_alias(name, alias)
    await ctx.asend_reply_msg(f'画廊\"{name}\"删除别名\"{alias}\"成功')


gall_pick = CmdHandler([
    '/gall pick', '/看',
], logger)
gall_pick.check_cdrate(cd).check_wblist(gbl)
@gall_pick.handle()
async def _(ctx: HandlerContext):
    limit = config.get('pick_limit')
    HELP = """
使用方式: 
/看 画廊名称 
/看 画廊名称 x2
/看 画廊名称 -1
/看 123 456...
""".strip()

    args = ctx.get_args().strip()
    if not args:
        raise ReplyException(HELP)

    pics, names = None, None
    try: 
        pids = [int(x) for x in args.split()]
        pics = [GalleryManager.get().find_pic(pid, raise_if_nofound=True) for pid in pids]
        names = [pic.gall_name for pic in pics]
    except: 
        pics = None
        num = 1
        args = args.replace('*', 'x').replace('×', 'x')
        if '-' in args:
            args, nindex_str = args.rsplit('-', 1)
            try: nindex = int(nindex_str)
            except: raise ReplyException(HELP)
            pics = GalleryManager.get().find_gall(args.strip(), raise_if_nofound=True).pics
            if len(pics) < nindex:
                raise ReplyException(f'画廊\"{args.strip()}\"仅有{len(pics)}张图片')
            pics = [pics[-nindex]]
        elif 'x' in args:
            args, num_str = args.rsplit('x', 1)
            try: num = int(num_str)
            except: raise ReplyException(HELP)
            assert_and_reply(1 <= num <= limit, f'一次查看图片数量必须在1到{limit}之间')
        names = [args.strip()]

    for name in names:
        g = GalleryManager.get().find_gall(name, raise_if_nofound=True)
        is_super = check_superuser(ctx.event)
        if not is_super and g.mode == GalleryMode.Off:
            raise ReplyException(f'画廊\"{name}\"已关闭')

    if pics is None:
        if not g.pics:
            raise ReplyException(f'画廊\"{name}\"没有图片')
        pics = [random.choice(g.pics) for _ in range(num)]

    if len(pics) > limit:
        raise ReplyException(f'一次最多查看{limit}张图片')

    await ctx.asend_msg(''.join([await get_image_cq(p.path, send_url_as_is=True) for p in pics]))


gall_add = CmdHandler([
    '/gall add', '/gall upload', '/上传', '/添加',
], logger)
gall_add.check_cdrate(cd).check_wblist(gbl)
@gall_add.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()

    check_duplicated = True
    if 'force' in args:
        check_duplicated = False
        args = args.replace('force', '').strip()

    name = args
    g = GalleryManager.get().find_gall(name, raise_if_nofound=True)

    is_super = check_superuser(ctx.event)
    if not is_super and g.mode != GalleryMode.Edit:
        raise ReplyException(f'画廊\"{name}\"不允许上传图片')

    await ctx.block(name)
    
    start_time = datetime.now()
    image_datas = await ctx.aget_image_datas()
    ok_list, err_msg = [], ""
    repeats: list[tuple[Image.Image, int]] = []
    REPEAT_IMAGE_SHOW_SIZE = (128, 128)

    async def download(url) -> str:
        async with TempDownloadFilePath(url, 'gif', remove_after=timedelta(minutes=10)) as p:
            return p
    paths = await batch_gather(*[download(data['url']) for data in image_datas], batch_size=16)

    for i, data in enumerate(image_datas, 1):
        try:
            path = paths[i-1]
            await run_in_pool(process_image_for_gallery, path, data.get('sub_type', 1))
            pid = await GalleryManager.get().async_add_pic(name, path, check_duplicated=check_duplicated)
            ok_list.append(pid)
            with open(ADD_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | @{ctx.event.user_id} upload pid={pid} to \"{name}\"\n")
        except GalleryPicRepeatedException as e:
            img = open_image(path)
            img.thumbnail(REPEAT_IMAGE_SHOW_SIZE)
            repeats.append((img, e.pid))
        except Exception as e:
            logger.print_exc(f"上传第{i}张图片到画廊\"{name}\"失败")
            err_msg += f"上传第{i}张图片失败: {get_exc_desc(e)}\n"
    cost_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"上传{len(ok_list)}/{len(image_datas)}张图片到画廊\"{name}\"完成, 耗时{cost_time:.2f}秒")

    hid = add_user_add_history(ctx.user_id, ok_list) if ok_list else None

    repeat_img = None
    if repeats:
        with Canvas(bg=FillBg(THUMBNAIL_BG_COLOR)).set_padding(8) as canvas:
            with VSplit().set_padding(0).set_sep(16).set_item_align('lt').set_content_align('lt'):
                TextBox(f"查重错误可使用\"/上传 force\"强制上传图片", TextStyle(DEFAULT_FONT, 16, BLACK))
                with Grid(row_count=int(math.sqrt(len(repeats) * 2)), hsep=8, vsep=8).set_item_align('t').set_content_align('t'):
                    for img, pid in repeats:
                        img2 = None
                        pic = GalleryManager.get().find_pic(pid, raise_if_nofound=False)
                        if pic and os.path.exists(pic.path):
                            img2 = open_image(pic.path)
                            img2.thumbnail(REPEAT_IMAGE_SHOW_SIZE)
                        with HSplit().set_padding(0).set_sep(4):
                            with VSplit().set_padding(0).set_sep(4).set_content_align('c').set_item_align('c'):
                                ImageBox(image=img, size=REPEAT_IMAGE_SHOW_SIZE, image_size_mode='fit').set_content_align('c')
                                TextBox(f"待上传图片", TextStyle(DEFAULT_FONT, 16, BLACK))
                            with VSplit().set_padding(0).set_sep(4).set_content_align('c').set_item_align('c'):
                                if img2:
                                    ImageBox(image=img2, size=REPEAT_IMAGE_SHOW_SIZE, image_size_mode='fit').set_content_align('c')
                                else:
                                    Spacer(w=REPEAT_IMAGE_SHOW_SIZE[0], h=REPEAT_IMAGE_SHOW_SIZE[1])
                                TextBox(f"pid: {pid}", TextStyle(DEFAULT_FONT, 16, BLACK))
                                
        repeat_img = await canvas.get_img()
    
    msg = f"[#{hid}] " if hid else ""
    msg += f"成功上传{len(ok_list)}/{len(image_datas)}张图片到\"{name}\"\n"
    msg += err_msg
    if repeats:
        msg += f"{len(repeats)}张图片与已有图片重复:"
        msg += await get_image_cq(repeat_img, low_quality=True)
    msg += "主要收录表情/梗图，请勿上传可能有争议的图片（使用\"/取消上传\"回退）"
    return await ctx.asend_fold_msg_adaptive(msg.strip())


gall_list = CmdHandler([
    '/gall list', '/看所有', "/看全部",
], logger)
gall_list.check_cdrate(cd).check_wblist(gbl)
@gall_list.handle()
async def _(ctx: HandlerContext):
    name = ctx.get_args().strip()
    
    # 列出所有画廊
    if not name:
        galls = GalleryManager.get().get_all_galls()
        if not galls:
            return await ctx.asend_reply_msg('当前没有任何画廊')
        
        with Canvas(bg=FillBg(THUMBNAIL_BG_COLOR)).set_padding(8) as canvas:
            with Grid(row_count=int(math.sqrt(len(galls))), hsep=8, vsep=8).set_item_align('t').set_content_align('t'):
                for name, g in galls.items():
                    cover: GalleryPic = GalleryManager.get().find_pic(g.cover_pid or 0)
                    if not cover and g.pics:
                        cover = g.pics[0]

                    total_size = 0
                    for p in glob.glob(os.path.join(g.pics_dir, '*')):
                        if os.path.isfile(p):
                            total_size += os.path.getsize(p)
                    total_size = total_size / (1024 * 1024)
                    if total_size == 0:
                        size_text = ""
                    elif total_size < 1:
                        size_text = "(<1M)"
                    elif total_size < 1024:
                        size_text = f"({total_size:.0f}M)"
                    else:
                        size_text = f"({total_size/1024:.0f}G)"

                    with VSplit().set_padding(0).set_sep(4).set_content_align('c').set_item_align('c'):
                        if cover:
                            await run_in_pool(cover.ensure_thumb)
                            ImageBox(image=open_image(cover.thumb_path), 
                                     size=(THUMBNAIL_SIZE[0]*2, THUMBNAIL_SIZE[1]*2), image_size_mode='fit').set_content_align('c')
                        else:
                            Spacer(w=THUMBNAIL_SIZE[0]*2, h=THUMBNAIL_SIZE[1]*2)
                        with HSplit().set_padding(0).set_sep(2).set_content_align('c').set_item_align('c'):
                            TextBox(f"{name}", TextStyle(DEFAULT_BOLD_FONT, 24, BLACK))
                            if g.mode != GalleryMode.Edit:
                                TextBox(f"[{g.mode.value}]", TextStyle(DEFAULT_FONT, 20, BLACK))

                        text = f"{len(g.pics)}张 {size_text}"
                        TextBox(text, TextStyle(DEFAULT_FONT, 20, BLACK))
                        TextBox(f"别名: {', '.join(g.aliases) if g.aliases else '无'}", TextStyle(DEFAULT_FONT, 12, (50, 50, 50)), use_real_line_count=True) \
                            .set_w(THUMBNAIL_SIZE[0] * 2).set_content_align('c')
                        
        return await ctx.asend_msg(
            await get_image_cq(
                await canvas.get_img(),
                low_quality=True,
            )
        )
    
    # 列出指定画廊的图片
    g = GalleryManager.get().find_gall(name, raise_if_nofound=True)
    is_super = check_superuser(ctx.event)
    if not is_super and g.mode == GalleryMode.Off:
        raise ReplyException(f'画廊\"{name}\"已关闭')
    
    assert_and_reply(g.pics, f'画廊\"{name}\"没有图片')
    
    with Canvas(bg=FillBg((230, 240, 255, 255))).set_padding(8) as canvas:
        with Grid(row_count=int(math.sqrt(len(g.pics))), hsep=4, vsep=4):
            for pic in g.pics:
                await run_in_pool(pic.ensure_thumb)
                with VSplit().set_padding(0).set_sep(2).set_content_align('c').set_item_align('c'):
                    if pic.thumb_path and os.path.exists(pic.thumb_path):
                        ImageBox(pic.thumb_path, size=THUMBNAIL_SIZE, image_size_mode='fit').set_content_align('c')
                    else:
                        Spacer(w=THUMBNAIL_SIZE[0], h=THUMBNAIL_SIZE[1])
                    TextBox(f"{pic.pid}", TextStyle(DEFAULT_FONT, 12, BLACK))

    return await ctx.asend_reply_msg(
        await get_image_cq(
            await canvas.get_img(),
            low_quality=True,
        )
    )


gall_log = CmdHandler([
    '/gall log',
], logger)
gall_log.check_cdrate(cd).check_wblist(gbl).check_superuser()
@gall_log.handle()
async def _(ctx: HandlerContext):
    try: 
        pid = int(ctx.get_args().strip())
    except:
        raise ReplyException('使用方式: /gall log pid')

    lines = []
    if os.path.exists(ADD_LOG_FILE):
        with open(ADD_LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
    for line in lines:
        if f'pid={pid}' in line:
            return await ctx.asend_reply_msg(line.strip())
        
    raise ReplyException(f'pid={pid}的上传记录不存在')
    

gall_download = CmdHandler([
    '/gall download', 
], logger)
gall_download.check_cdrate(cd).check_wblist(gbl).check_superuser().check_group()
@gall_download.handle()
async def _(ctx: HandlerContext):
    name = ctx.get_args().strip()
    g = GalleryManager.get().find_gall(name, raise_if_nofound=True)
    assert_and_reply(g.pics, f'画廊\"{name}\"没有图片')

    with TempFilePath(".zip", remove_after=timedelta(minutes=5)) as zip_path:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for pic in g.pics:
                arcname = os.path.basename(pic.path)
                zipf.write(pic.path, arcname)
        
        filesize = os.path.getsize(zip_path) / (1024 * 1024)
        await ctx.asend_reply_msg(f'正在发送画廊\"{name}\"所有{len(g.pics)}张图片的压缩包({filesize:.2f}M)...')

        await upload_group_file(
            ctx.bot,
            ctx.group_id,
            zip_path,
            f"{name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        )


gall_cover = CmdHandler([
    '/gall cover', 
], logger)
gall_cover.check_cdrate(cd).check_wblist(gbl).check_superuser()
@gall_cover.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip().split()
    if len(args) != 2:
        raise ReplyException('使用方式: /gall cover 画廊名称 图片ID')
    name, pid_str = args
    try:
        pid = int(pid_str)
    except:
        raise ReplyException('图片ID必须是整数')

    GalleryManager.get().set_cover_pic(name, pid)
    await ctx.asend_reply_msg(f'画廊\"{name}\"封面图片设置为pid={pid}成功')


gall_check = CmdHandler([
    '/gall check', 
], logger)
gall_check.check_cdrate(cd).check_wblist(gbl).check_superuser()
@gall_check.handle()
async def _(ctx: HandlerContext):
    name = ctx.get_args().strip()

    rehash = False
    if 'rehash' in name:
        rehash = True
        name = name.replace('rehash', '').strip()

    all = False
    if name == 'all':
        all = True
        name = name.replace('all', '').strip()

    async def check(name: str):
        if rehash:
            msg = f'正在为画廊\"{name}\"重新计算hash并检查重复图片...'
        else:
            msg = f'正在为画廊\"{name}\"检查重复图片...'
        await ctx.asend_reply_msg(msg)
        res = await GalleryManager.get().async_check_gallery(name, rehash=rehash)

        if not res:
            return await ctx.asend_reply_msg(f'画廊\"{name}\"重新计算hash完成，未发现重复图片')

        REPEAT_IMAGE_SHOW_SIZE = (128, 128)
        with Canvas(bg=FillBg(THUMBNAIL_BG_COLOR)).set_padding(8) as canvas:
            with VSplit().set_padding(16).set_sep(8).set_item_align('lt').set_content_align('lt'):
                for first_pid, repeat_pids in res.items():
                    pids = [first_pid] + repeat_pids
                    with HSplit().set_padding(0).set_sep(4).set_item_align('lt').set_content_align('lt'):
                        for pid in pids:
                            img = None
                            pic = GalleryManager.get().find_pic(pid, raise_if_nofound=False)
                            def open_img():
                                if pic and os.path.exists(pic.path):
                                    img = open_image(pic.path)
                                    img.thumbnail(REPEAT_IMAGE_SHOW_SIZE)
                                    return img
                                return None
                            img = await run_in_pool(open_img)
                            with VSplit().set_padding(0).set_sep(4).set_content_align('c').set_item_align('c'):
                                if img:
                                    ImageBox(image=img, size=REPEAT_IMAGE_SHOW_SIZE, image_size_mode='fit').set_content_align('c')
                                else:
                                    Spacer(w=REPEAT_IMAGE_SHOW_SIZE[0], h=REPEAT_IMAGE_SHOW_SIZE[1])
                                TextBox(f"pid: {pid}", TextStyle(DEFAULT_FONT, 16, BLACK))
        
        repeat_img = await canvas.get_img()
        if rehash:
            msg = f'画廊\"{name}\"重新计算hash完成，发现重复图片组共{len(res)}组:\n'
        else:
            msg = f'画廊\"{name}\"检查完成，发现重复图片组共{len(res)}组:\n'
        msg += await get_image_cq(repeat_img, low_quality=True)
        msg += f'推荐移除的重复图片pid:'
        for first_pid, repeat_pids in res.items():
            msg += '\n' + ' '.join(str(pid) for pid in repeat_pids)
        return await ctx.asend_fold_msg_adaptive(msg.strip())
    
    if all:
        galls = GalleryManager.get().get_all_galls()
        for name in galls.keys():
            await check(name)
    else:
        await check(name)


gall_replace = CmdHandler([
    '/gall replace', 
], logger)
gall_replace.check_cdrate(cd).check_wblist(gbl).check_superuser()
@gall_replace.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()

    try:
        check_duplicated = True
        if 'force' in args:
            check_duplicated = False
            args.remove('force')
        pid = int(args)
    except:
        raise ReplyException('使用方式: /gall replace pid')

    image_data = await ctx.aget_image_datas(return_first=True, max_count=1)
    if not image_data:
        raise ReplyException('请附加要替换的图片')
    
    async with TempDownloadFilePath(image_data['url'], 'gif') as path:
        await run_in_pool(process_image_for_gallery, path, image_data.get('sub_type', 1))
        try:
            pid = await GalleryManager.get().async_replace_pic(pid, path, check_duplicated=check_duplicated)
        except GalleryPicRepeatedException as e:
            raise ReplyException(f'替换失败: 画廊中已存在相似图片(pid={e.pid})')

    await ctx.asend_reply_msg(f'成功替换图片pid={pid}')


gall_download_all = CmdHandler([
    '/gall download link', '/下载图包', '/下载看', '/下载画廊',
], logger, priority=1)
gall_download_all.check_cdrate(cd).check_wblist(gbl)
@gall_download_all.handle()
async def _(ctx: HandlerContext):
    return await ctx.asend_msg(config.get('sync.share_link'))


gall_cancel = CmdHandler([
    '/gall cancel', '/gall revert', '/取消上传', '/撤销上传', '/回退上传',
], logger)
gall_cancel.check_cdrate(cd).check_wblist(gbl)
@gall_cancel.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()

    hid = None
    if args:
        try:
            hid = int(args)
        except:
            raise ReplyException(f"""
使用方式:
撤销你的最近一次上传: 
{ctx.trigger_cmd}  
撤销指定上传记录(仅管理): 
{ctx.trigger_cmd} 记录ID
""".strip())
    
    if hid:
        assert_and_reply(check_superuser(ctx.event), '仅管理员可撤销指定上传记录，非管理员可留空参数撤销自己的最近一次上传')
        h, ok_list, err_list = revert_user_add_history(hid)
        msg = f"撤销{h['uid']}的上传记录#{h['id']}\n"
    else:
        h, ok_list, err_list = revert_user_last_add_history(ctx.user_id)
        msg = f"撤销你的最近一次上传记录#{h['id']}\n"

    if ok_list:
        msg += f'{len(ok_list)}张图片删除成功:\n'
        msg += ' '.join(str(pid) for pid in ok_list) + '\n'
    if err_list:
        msg += f'{len(err_list)}张图片删除失败:\n'
        msg += ' '.join(str(pid) for pid in err_list) + '\n'
    await ctx.asend_fold_msg_adaptive(msg.strip())


gall_record = CmdHandler([
    '/gall record', '/上传记录',
], logger)
gall_record.check_cdrate(cd).check_wblist(gbl)
@gall_record.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()
    try:
        hid = int(args)
    except:
        raise ReplyException(f'使用方式: {ctx.trigger_cmd} 记录ID')
    h = get_user_add_history(hid)
    user_id = h['uid']
    time = datetime.fromtimestamp(h['ts']).strftime('%Y-%m-%d %H:%M:%S')
    pids = h['pids']
    reverted = h['reverted']

    pics: list[GalleryPic] = []
    no_found_pids: list[int] = []
    for pid in pids:
        if pic := GalleryManager.get().find_pic(pid, raise_if_nofound=False):
            pics.append(pic)
        else:
            no_found_pids.append(pid)

    msg = f"{user_id}的上传记录#{h['id']}\n"
    msg += f"{time}\n"
    if reverted:
        msg += f"该上传已撤销\n"
    msg += f"上传的图片数量:{len(pids)}\n"
    if no_found_pids:
        msg += f"未找到的图片id: {' '.join(str(pid) for pid in no_found_pids)}\n"
    if pics:
        with Canvas(bg=FillBg(THUMBNAIL_BG_COLOR)).set_padding(8) as canvas:
            with Grid(row_count=int(math.sqrt(len(pics))), hsep=4, vsep=4):
                for pic in pics:
                    await run_in_pool(pic.ensure_thumb)
                    with VSplit().set_padding(0).set_sep(2).set_content_align('c').set_item_align('c'):
                        if pic.thumb_path and os.path.exists(pic.thumb_path):
                            ImageBox(pic.thumb_path, size=THUMBNAIL_SIZE, image_size_mode='fit').set_content_align('c')
                        else:
                            Spacer(w=THUMBNAIL_SIZE[0], h=THUMBNAIL_SIZE[1])
                        TextBox(f"{pic.pid}", TextStyle(DEFAULT_FONT, 12, BLACK))
        msg += await get_image_cq(
            await canvas.get_img(),
            low_quality=True,
        )

    return await ctx.asend_fold_msg_adaptive(msg)


# ======================= 定时任务 ======================= # 

# 自动同步画廊图片到百度网盘
for sync_time in config.get("sync.sync_times"):
    @scheduled_job("cron", hour=int(sync_time[0]), minute=int(sync_time[1]), second=int(sync_time[2]))
    async def sync_bypy():
        if not config.get('sync.enable'):
            return
        remote_dir = config.get('sync.remote_dir')
        local_dir = f"data/gallery/tmp/"
        verbose = config.get('sync.verbose')

        def sync(g: Gallery):
            try:
                logger.info(f'开始同步画廊\"{g.name}\"到百度网盘({remote_dir})')
                for p in g.pics:
                    if os.path.exists(p.path):
                        _, ext = os.path.splitext(os.path.basename(p.path))
                        dst = os.path.join(local_dir, g.name, f"{p.pid}{ext}")
                        create_parent_folder(dst)
                        shutil.copy2(p.path, dst)
                
                command = [
                    'bypy',
                    'syncup',
                    os.path.join(local_dir, g.name),
                    os.path.join(remote_dir, g.name),
                    'True', '-v'
                ]
                process = subprocess.Popen(
                    command, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    encoding='utf-8'
                )
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output and verbose:
                        logger.info(f"[bypy] {output.strip()}")
                if process.returncode != 0:
                    raise Exception(f'bypy执行失败: code={process.returncode}')
                
                logger.info(f'画廊\"{g.name}\"同步完成')
            except Exception as e:
                logger.error(f'同步画廊\"{g.name}\"失败: {get_exc_desc(e)}')

        for g in GalleryManager.get().galleries.values():
            await run_in_pool(sync, g)
            remove_folder(local_dir)
        