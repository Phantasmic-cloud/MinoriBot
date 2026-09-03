# 用于 MinoriBot 附属服务的通用 Utils
import os
import os.path as osp
from os.path import join as pjoin
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Union
from datetime import datetime, timedelta
import sys
import yaml
from copy import deepcopy
import asyncio
import traceback
import orjson
import random
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


SERVICE_NAME = "MinoriBot-autochat"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
CONFIG_DIR = os.path.join(_ROOT, "config")
CONFIG_NAME = "chat.autochat"
NEED_RPC = True


import setproctitle
setproctitle.setproctitle(SERVICE_NAME)


# ========================== Utils ========================== #

def count_dict(d: dict, level: int) -> int:
    """
    计算字典某个层级的元素个数
    """
    if level == 1:
        return len(d)
    else:
        return sum(count_dict(v, level-1) for v in d.values())

class Counter:
    def __init__(self):
        self.count = {}
    def inc(self, key, value=1):
        self.count[key] = self.count.get(key, 0) + value
    def get(self, key):
        return self.count.get(key, 0)
    def items(self):
        return self.count.items()
    def keys(self):
        return self.count.keys()
    def values(self):
        return self.count.values()
    def __len__(self):
        return len(self.count)
    def __str__(self):
        return str(self.count)
    def clear(self):
        self.count.clear()
    def __getitem__(self, key):
        return self.count.get(key, 0)
    def __setitem__(self, key, value):
        self.count[key] = value
    def keys(self):
        return self.count.keys()

def find_by(lst: list[dict[str, Any]], key: str, value: Any, mode="first", use_str_compare=False):
    """
    用某个key查找某个dict列表中的元素 mode=first/last/all
    查找单个时找不到返回None, 查找多个时找不到返回空列表
    """
    # 检查是否出现类型不匹配
    if lst and key in lst[0]:
        if type(lst[0][key]) != type(value):
            warning(f"find_by 发现类型不匹配:\n{traceback.format_stack()}")

    if mode == "first":
        for item in lst:
            if key in item:
                if use_str_compare:
                    if str(item[key]) == str(value):
                        return item
                else:
                    if item[key] == value:
                        return item
        return None
    elif mode == "last":
        for item in reversed(lst):
            if key in item:
                if use_str_compare:
                    if str(item[key]) == str(value):
                        return item
                else:
                    if item[key] == value:
                        return item
        return None
    elif mode == "all":
        if use_str_compare:
            return [item for item in lst if key in item and str(item[key]) == str(value)]
        else:
            return [item for item in lst if key in item and item[key] == value]
    else:
        raise Exception("find_by mode must be first/last/all")

def unique_by(lst: list[dict[str, Any]], key: str):
    """
    获取按某个key去重后的dict列表
    """
    val_set = set()
    ret = []
    for item in lst:
        if item[key] not in val_set:
            val_set.add(item[key])
            ret.append(item)
    return ret

def unique_idx_by(lst: list[dict[str, Any]], key: str) -> list[int]:
    """
    获取按某个key去重后的dict列表，返回索引
    """
    val_set = set()
    ret = []
    for idx, item in enumerate(lst):
        if item[key] not in val_set:
            val_set.add(item[key])
            ret.append(idx)
    return ret

def remove_by(lst: list[dict[str, Any]], key: str, value: Any):
    """
    获取删除某个key为某个值的所有项的dict列表
    """
    return [item for item in lst if key not in item or item[key] != value]

def find_by_predicate(lst: list[Any], predicate, mode="first"):
    """
    用某个条件查找某个列表中的元素 mode=first/last/all
    查找单个时找不到返回None, 查找多个时找不到返回空列表
    """
    if mode not in ["first", "last", "all"]:
        raise Exception("find_by_func mode must be first/last/all")
    ret = [item for item in lst if predicate(item)]
    if not ret: 
        return None if mode != "all" else []
    if mode == "first":
        return ret[0]
    if mode == "last":
        return ret[-1]
    return ret

def unique_by_predicate(lst: list[Any], predicate):
    """
    获取按某个条件去重后的dict列表
    """
    val_set = set()
    ret = []
    for item in lst:
        if predicate(item) not in val_set:
            val_set.add(predicate(item))
            ret.append(item)
    return ret

def remove_by_predicate(lst: list[Any], predicate):
    """
    获取删除某个条件的dict列表
    """
    return [item for item in lst if not predicate(item)]

def load_json(file_path: str) -> dict:
    with open(file_path, 'rb') as file:
        return orjson.loads(file.read())
    
def dump_json(data: dict, file_path: str, indent: bool = True) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, 'wb') as file:
        buffer = orjson.dumps(data, option=orjson.OPT_INDENT_2 if indent else 0)
        file.write(buffer)
    os.replace(tmp_path, file_path)

def loads_json(s: str | bytes) -> dict:
    return orjson.loads(s)

def dumps_json(data: dict, indent: bool = True) -> str:
    return orjson.dumps(data, option=orjson.OPT_INDENT_2 if indent else 0).decode('utf-8')

async def aload_json(path: str) -> dict[str, Any]:
    return await asyncio.to_thread(load_json, path)

async def asave_json(data: dict[str, Any], path: str):
    return await asyncio.to_thread(dump_json, data, path)

def get_exc_desc(e: Exception) -> str:
    et = type(e).__name__
    e = str(e)
    if et in ['AssertionError', 'HTTPException', 'Exception']:
        return e
    if et and e:
        return f"{et}: {e}"
    return et or e

def create_parent_folder(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def get_md5(s: Union[str, bytes]) -> str:
    import hashlib
    m = hashlib.md5()
    if isinstance(s, str): s = s.encode()
    m.update(s)
    return m.hexdigest()

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串之间的Levenshtein距离
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def get_readable_file_size(size: int) -> str:
    """
    将文件大小(byte)转换为可读字符串
    """
    if size < 1024:
        return f"{size}B"
    size /= 1024
    if size < 1024:
        return f"{size:.2f}KB"
    size /= 1024
    if size < 1024:
        return f"{size:.2f}MB"
    size /= 1024
    return f"{size:.2f}GB"

def get_readable_datetime(t: datetime | float | int, show_original_time=True, use_en_unit=False):
    """
    将时间点转换为可读字符串
    """
    if isinstance(t, float) or isinstance(t, int):
        t = datetime.fromtimestamp(t)
    day_unit, hour_unit, minute_unit, second_unit = ("天", "小时", "分钟", "秒") if not use_en_unit else ("d", "h", "m", "s")
    now = datetime.now()
    diff = t - now
    text, suffix = "", "后"
    if diff.total_seconds() < 0:
        suffix = "前"
        diff = -diff
    if diff.total_seconds() < 60:
        text = f"{int(diff.total_seconds())}{second_unit}"
    elif diff.total_seconds() < 60 * 60:
        text = f"{int(diff.total_seconds() / 60)}{minute_unit}"
    elif diff.total_seconds() < 60 * 60 * 24:
        text = f"{int(diff.total_seconds() / 60 / 60)}{hour_unit}{int(diff.total_seconds() / 60 % 60)}{minute_unit}"
    else:
        text = f"{diff.days}{day_unit}"
    text += suffix
    if show_original_time:
        text = f"{t.strftime('%Y-%m-%d %H:%M:%S')} ({text})"
    return text

def get_readable_timedelta(delta: timedelta, precision: str = 'm', use_en_unit=False) -> str:
    """
    将时间段转换为可读字符串
    """
    match precision:
        case 's': precision = 3
        case 'm': precision = 2
        case 'h': precision = 1
        case 'd': precision = 0

    s = int(delta.total_seconds())
    if s <= 0: return "0秒" if not use_en_unit else "0s"
    d = s // (24 * 3600)
    s %= (24 * 3600)
    h = s // 3600
    s %= 3600
    m = s // 60
    s %= 60

    ret = ""
    if d > 0: 
        ret += f"{d}天" if not use_en_unit else f"{d}d"
    if h > 0 and (precision >= 1 or not ret): 
        ret += f"{h}小时" if not use_en_unit else f"{h}h"
    if m > 0 and (precision >= 2 or not ret):
        ret += f"{m}分钟" if not use_en_unit else f"{m}m"
    if s > 0 and (precision >= 3 or not ret):
        ret += f"{s}秒"   if not use_en_unit else f"{s}s"
    return ret

def truncate(s: str, limit: int) -> str:
    """
    截断字符串到指定长度，中文字符算两个字符
    """
    s = str(s)
    if s is None: return "<None>"
    l = 0
    for i, c in enumerate(s):
        if l >= limit:
            return s[:i] + "..."
        l += 1 if ord(c) < 128 else 2
    return s

def get_str_display_length(s: str) -> int:
    """
    获取字符串的显示长度，中文字符算两个字符
    """
    l = 0
    for c in s:
        l += 1 if ord(c) < 128 else 2
    return l

def get_str_line_count(s: str, line_length: int) -> int:
    """
    获取字符串在指定行长度下的行数
    """
    lines = [""]
    for c in s:
        if c == '\n':
            lines.append("")
            continue
        if get_str_display_length(lines[-1] + c) > line_length:
            lines.append("")
        lines[-1] += c
    return len(lines)

def get_float_str(value: float, precision: int = 2, remove_zero: bool = True) -> str:
    """
    将浮点数转换为字符串，保留指定小数位数，并可选择去除末尾的零
    """
    ret = f"{value:.{precision}f}"
    if remove_zero:
        ret = ret.rstrip('0').rstrip('.')
    return ret


# ========================== Config ========================== #

@dataclass
class ConfigData:
    mtime: int
    data: dict = field(default_factory=dict)

class ConfigItem:
    """
    配置项类，用于动态延迟获取配置文件中的单个配置项
    """
    def __init__(self, config: 'Config', key: str):
        self.config = config
        self.key = key

    def get(self, default=None, raise_exc: Optional[bool]=None) -> Any:
        return self.config.get(self.key, default, raise_exc)
    
class Config:
    _data: dict[str, ConfigData] = {}

    def __init__(self, name: str):
        """
        初始化配置类
        name: 配置名称，格式为 "module" 或 "module.submodule"
        """
        self.name = name
        self.path = pjoin(CONFIG_DIR, name.replace('.', '/') + '.yaml')
        
    def _update(self):
        if not osp.exists(self.path):
            print(f"[WARNING] 找不到配置文件 {self.path}")
            # raise FileNotFoundError(f"配置文件 {self.path} 不存在")
            return
        mtime = int(os.path.getmtime(self.path))
        if self.name not in Config._data or Config._data[self.name].mtime != mtime:
            try:
                with open(self.path, 'r', encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                Config._data[self.name] = ConfigData(mtime=mtime, data=data)
            except Exception as e:
                print(f"[WARNING] 读取配置文件 {self.path} 失败: {e}")
                Config._data[self.name].mtime = mtime  # 避免重复读取

    def get_all(self) -> dict:
        """
        获取配置项的所有数据
        """
        self._update()
        return deepcopy(Config._data.get(self.name, ConfigData(0, {})).data)

    def get(self, key: str, default=None, raise_exc: Optional[bool]=None) -> Any:
        """
        获取配置项的值
        key: 配置项的键，格式为 "key" 或 "key1.key2"
        default: 如果配置项不存在返回的默认值
        raise_exc: 如果配置项不存在，是否抛出异常，为None时如果default为None则抛出异常，否则返回default
        """
        if raise_exc is None:
            raise_exc = default is None
        self._update()
        if isinstance(key, str):
            keys = key.split('.')
        else:
            keys = [key]
        ret = Config._data.get(self.name, ConfigData(0, {})).data
        for k in keys:
            if k not in ret:
                if raise_exc:
                    raise KeyError(f"配置 {self.name} 中不存在 {key}")
                return default
            ret = ret[k]
        return deepcopy(ret)
    
    def mtime(self) -> int:
        """
        获取配置文件的修改时间
        """
        self._update()
        return Config._data.get(self.name, ConfigData(0, {})).mtime
    
    def item(self, key: str) -> ConfigItem:
        """
        获取配置项的延迟加载对象
        key: 配置项的键，格式为 "key" 或 "key1.key2"
        """
        return ConfigItem(self, key)
    
def get_cfg_or_value(obj: Union[ConfigItem, Any], default=None, raise_exc: Optional[bool]=None) -> Any:
    """
    如果是 ConfigItem 对象则返回值，否则返回原对象
    """
    if isinstance(obj, ConfigItem):
        return obj.get(default, raise_exc)
    return obj

def parse_cfg_num(x: str) -> Union[int, float]:
    """
    解析配置中的数字字符串，支持数字和数字四则运算
    """
    if isinstance(x, (int, float)):
        return x
    try:
        return eval(x, {'__builtins__': None}, {})
    except Exception as e:
        raise ValueError(f"无法解析配置数字 '{x}': {e}")

if CONFIG_NAME:
    config = Config(CONFIG_NAME)


# ========================== Log ========================== #

LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR']

def log(level: str, *args, **kwargs):
    log_level = config.get('log_level').upper()
    if LOG_LEVELS.index(log_level) > LOG_LEVELS.index(level):
        return
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}][{level}]', *args, **kwargs, flush=True)

def debug(*args, **kwargs):
    log('DEBUG', *args, **kwargs)

def info(*args, **kwargs):
    log('INFO', *args, **kwargs)

def warning(*args, **kwargs):
    log('WARNING', *args, **kwargs)

def error(*args, print_exc: bool = True, **kwargs):
    log('ERROR', *args, **kwargs)
    if print_exc:
        traceback.print_exc()
    

# ========================== FileDB ========================== #

class FileDB:
    def __init__(self, path: str):
        self.path = path
        self.data = {}
        self.load()

    def load(self):
        try:
            self.data = load_json(self.path)
            debug(f'加载数据库 {self.path} 成功')
        except:
            debug(f'加载数据库 {self.path} 失败 使用空数据')
            self.data = {}

    def keys(self) -> set[str]:
        return self.data.keys()

    def save(self):
        dump_json(self.data, self.path)
        debug(f'保存数据库 {self.path}')

    def get(self, key: str, default: Any=None) -> Any:
        """
        - 获取某个key的值，找不到返回default
        - 直接返回缓存对象，若要进行修改又不影响DB内容则必须自行deepcopy
        """
        assert isinstance(key, str), f'key必须是字符串，当前类型: {type(key)}'
        return self.data.get(key, default)

    def get_copy(self, key: str, default: Any=None) -> Any:
        assert isinstance(key, str), f'key必须是字符串，当前类型: {type(key)}'
        return deepcopy(self.data.get(key, default))

    def set(self, key: str, value: Any):
        assert isinstance(key, str), f'key必须是字符串，当前类型: {type(key)}'
        debug(f'设置数据库 {self.path} {key} = {truncate(str(value), 32)}')
        self.data[key] = deepcopy(value)
        self.save()

    def delete(self, key: str):
        assert isinstance(key, str), f'key必须是字符串，当前类型: {type(key)}'
        debug(f'删除数据库 {self.path} {key}')
        if key in self.data:
            del self.data[key]
            self.save()

_file_dbs: dict[str, FileDB] = {}
def get_file_db(path: str) -> FileDB:
    global _file_dbs
    if path not in _file_dbs:
        _file_dbs[path] = FileDB(path)
    return _file_dbs[path]


# ========================== Rpc ========================== #

if NEED_RPC:
    import websockets
    import aiorpcx

    class RpcNotConnectedError(Exception):
        pass

    class RpcSession:
        def __init__(
            self, 
            host: str | ConfigItem, 
            port: int | ConfigItem, 
            token: str | ConfigItem,
            reconnect_interval: int | ConfigItem,
        ):
            self.host = host
            self.port = port
            self.token = token
            self.reconnect_interval = reconnect_interval
            self.session = None
            self.ws_client = None
        
        def is_connected(self):
            return self.session is not None

        async def connect(self):
            host = get_cfg_or_value(self.host)
            port = get_cfg_or_value(self.port)
            ws = aiorpcx.connect_ws(f'ws://{host}:{port}')
            self.session: aiorpcx.RPCSession = await ws.__aenter__()
            self.session.sent_request_timeout = 10000
            self.ws_client = ws
            info(f"成功连接到RPC服务器 {host}:{port}")
        
        async def close(self):
            if self.session:
                await self.ws_client.__aexit__(None, None, None)
                self.session = None
                self.ws_client = None
                info("RPC连接已关闭")
        
        async def call(
            self, 
            method: str, 
            *args,
            timeout: int = config.get('rpc.default_timeout'),
        ):
            if not self.is_connected():
                raise RpcNotConnectedError()
            args_with_token = [get_cfg_or_value(self.token)] + list(args)
            try:
                debug(f"发送RPC请求: {method} {args}")
                return await asyncio.wait_for(self.session.send_request(method, args_with_token), get_cfg_or_value(timeout))
            except websockets.exceptions.ConnectionClosed as e:
                debug(f"RPC连接已关闭: {method} {args} -> {get_exc_desc(e)}")
                self.session = None
                raise Exception("RPC连接已关闭")
            except aiorpcx.RPCError as e:
                debug(f"RPC请求错误: {method} {args} -> {get_exc_desc(e)}")
                raise e

        async def run(self, reconnect: bool):
            while True:
                reconn_interval = get_cfg_or_value(self.reconnect_interval)
                try:
                    if not self.is_connected():
                        await self.connect()
                        if not reconnect:
                            break
                except Exception as e:
                    warning(f"连接RPC服务器失败: {get_exc_desc(e)}，{reconn_interval}秒后重试")
                finally:
                    await asyncio.sleep(reconn_interval)

