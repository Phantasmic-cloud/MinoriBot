from .common import *
import math
import io
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Union, Any, Set
from dataclasses import dataclass
from PIL import Image

from src.draw.painter import resize_keep_ratio

asset_config = Config("pjsk.asset")
ASSET_DEBUG_CFG = asset_config.item("debug")


def get_image_pixel_hash(img: Image.Image):
    return hashlib.md5(img.tobytes()).hexdigest()


# ================================ MasterData资源 ================================ #

DEFAULT_VERSION = "0.0.0.0"
MASTER_DB_CACHE_DIR = f"{SEKAI_ASSET_DIR}/masterdata/"
DEFAULT_INDEX_KEYS = ["id"]
DEFAULT_SORT_KEYS = []


def get_multi_keys(data: dict, keys: List[Any]):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of the keys {keys} found in dict")


def get_version_order(version: str) -> tuple:
    parts = []
    for chunk in str(version or "0").replace("-", ".").split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    return tuple(parts) if parts else (0,)


@dataclass
class RegionMasterDbSource:
    name: str
    base_url: str
    version_url: str
    version: str = DEFAULT_VERSION
    asset_version: str = DEFAULT_VERSION

    async def update_version(self):
        version = DEFAULT_VERSION
        try:
            timeout = asset_config.get("default_masterdata_update_check_timeout")
            version_data = await asyncio.wait_for(download_json(self.version_url), timeout)
            version = version_data.get("cdnVersion", 0)
            if not version:
                version = get_multi_keys(version_data, ["data_version", "dataVersion"])
            self.version = str(version)
            self.asset_version = get_multi_keys(version_data, ["asset_version", "assetVersion"])
        except asyncio.TimeoutError:
            logger.error("获取 MasterDB [%s] 的版本信息超时", self.name)
        except Exception:
            logger.print_exc(f"获取 MasterDB [{self.name}] 的版本信息失败")


class RegionMasterDbManager:
    _all_mgrs = {}
    _update_hooks = []

    def __init__(self, region: str, sources: List[RegionMasterDbSource], version_update_interval: timedelta):
        self.region = region
        self.sources = sources
        self.latest_source = None
        self.version_update_interval = version_update_interval
        self.version_update_time = None

    async def update(self):
        last_version = self.latest_source.version if self.latest_source else DEFAULT_VERSION
        last_asset_version = self.latest_source.asset_version if self.latest_source else DEFAULT_VERSION
        await asyncio.gather(*[source.update_version() for source in self.sources])
        self.sources.sort(key=lambda x: get_version_order(x.version), reverse=True)
        self.latest_source = self.sources[0]
        self.version_update_time = datetime.now()
        if last_version != DEFAULT_VERSION and last_version != self.latest_source.version:
            logger.info(
                "获取到最新版本的 MasterDB [%s.%s] 版本为 %s",
                self.region, self.latest_source.name, self.latest_source.version,
            )
            for hook in self._update_hooks:
                asyncio.create_task(hook(
                    self.region, self.latest_source.name,
                    self.latest_source.version, last_version,
                    self.latest_source.asset_version, last_asset_version,
                ))

    async def get_latest_source(self) -> RegionMasterDbSource:
        if not self.latest_source or datetime.now() - self.version_update_time > self.version_update_interval:
            await self.update()
        return self.latest_source

    async def get_all_sources(self, force_update=False) -> List[RegionMasterDbSource]:
        if force_update or not self.latest_source or datetime.now() - self.version_update_time > self.version_update_interval:
            await self.update()
        return self.sources

    @classmethod
    def on_update(cls):
        def _wrapper(func):
            cls._update_hooks.append(func)
            return func
        return _wrapper

    @classmethod
    def get(cls, region: str) -> "RegionMasterDbManager":
        if region not in cls._all_mgrs:
            cfg = asset_config.get_all()
            assert region in cfg and "masterdata" in cfg[region], f"未找到 {region} 的 MasterData 配置"
            region_config = cfg[region]["masterdata"]
            cls._all_mgrs[region] = RegionMasterDbManager(
                region=region,
                sources=[RegionMasterDbSource(**source) for source in region_config["sources"]],
                version_update_interval=timedelta(minutes=region_config.get("version_update_interval", 10)),
            )
        return cls._all_mgrs[region]


class MasterDataManager:
    _all_mgrs = {}

    def __init__(self, name: str, build_indices: Optional[bool] = None):
        self.name = name
        self.version = {}
        self.data: Dict[str, Any] = {}
        self.update_hooks = []
        self.map_fn = {}
        self.download_fn = {}
        self._set_index_keys(DEFAULT_INDEX_KEYS)
        self.indexed_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._set_sort_keys(DEFAULT_SORT_KEYS)
        self.sorted_data: Dict[str, Dict[str, List[Any]]] = {}
        self.lock = asyncio.Lock()

    def _set_index_keys(self, index_keys: Union[str, List[str], Dict[str, List[str]]]):
        if isinstance(index_keys, str):
            index_keys = [index_keys]
        if isinstance(index_keys, list):
            index_keys = {region: index_keys for region in ALL_SERVER_REGIONS}
        self.index_keys = index_keys

    def _set_sort_keys(self, sort_keys: Union[str, List[str], Dict[str, List[str]]]):
        if isinstance(sort_keys, str):
            sort_keys = [sort_keys]
        if isinstance(sort_keys, list):
            sort_keys = {region: sort_keys for region in ALL_SERVER_REGIONS}
        self.sort_keys = sort_keys

    def get_cache_path(self, region: str) -> str:
        create_folder(pjoin(MASTER_DB_CACHE_DIR, region))
        return pjoin(MASTER_DB_CACHE_DIR, region, f"{self.name}.json")

    def _build_indexed_data(self, region: str):
        try:
            if self.data.get(region, None) is None:
                logger.warning("MasterData [%s.%s] 构建索引发生在数据加载前", region, self.name)
                return
            if not self.data[region] or not isinstance(self.data[region], list):
                return
            self.indexed_data[region] = {}
            for key in self.index_keys.get(region, []):
                ind = {}
                for item in self.data[region]:
                    if key not in item:
                        continue
                    ind.setdefault(item[key], []).append(item)
                if ind:
                    self.indexed_data[region][key] = ind
        except Exception:
            logger.print_exc(f"MasterData [{region}.{self.name}] 构建索引失败")

    def _build_sorted_data(self, region: str):
        try:
            if self.data.get(region, None) is None:
                logger.warning("MasterData [%s.%s] 构建排序发生在数据加载前", region, self.name)
                return
            if not self.data[region] or not isinstance(self.data[region], list):
                return
            self.sorted_data[region] = {}
            for key in self.sort_keys.get(region, []):
                sorted_list = sorted(self.data[region], key=lambda x: x.get(key))
                if sorted_list:
                    self.sorted_data[region][key] = sorted_list
        except Exception:
            logger.print_exc(f"MasterData [{region}.{self.name}] 构建排序失败")

    async def _load_from_cache(self, region: str):
        cache_path = self.get_cache_path(region)
        assert os.path.exists(cache_path), "缓存不存在"
        versions = file_db.get_copy("master_data_cache_versions", {}).get(region, {})
        assert self.name in versions, "缓存版本无效"
        self.version[region] = versions[self.name]
        self.data[region] = await aload_json(cache_path)
        logger.info("MasterData [%s.%s] 从本地加载成功", region, self.name)
        map_fn = self.map_fn.get("all", self.map_fn.get(region))
        if map_fn:
            self.data[region] = await run_in_pool(map_fn, self.data[region])
        await run_in_pool(self._build_indexed_data, region)
        await run_in_pool(self._build_sorted_data, region)

    async def _download_from_db(self, region: str, source: RegionMasterDbSource):
        cache_path = self.get_cache_path(region)
        if not source.base_url.endswith("/"):
            source.base_url += "/"
        url = f"{source.base_url}{self.name}.json"
        download_fn = self.download_fn.get("all", self.download_fn.get(region))
        timeout = asset_config.get("default_masterdata_download_timeout")

        async def _download():
            if not download_fn:
                self.data[region] = await download_json(url)
            else:
                self.data[region] = await download_fn(source.base_url)

        try:
            await asyncio.wait_for(_download(), timeout)
        except asyncio.TimeoutError:
            logger.warning("下载 MasterData [%s.%s] 超时", region, self.name)
            return
        self.version[region] = source.version
        versions = file_db.get("master_data_cache_versions", {})
        if region not in versions:
            versions[region] = {}
        versions[region][self.name] = self.version[region]
        file_db.set("master_data_cache_versions", versions)
        await adump_json(self.data[region], cache_path)
        logger.info("MasterData [%s.%s] 更新成功", region, self.name)
        map_fn = self.map_fn.get("all", self.map_fn.get(region))
        if map_fn:
            self.data[region] = await run_in_pool(map_fn, self.data[region])
        await run_in_pool(self._build_indexed_data, region)
        await run_in_pool(self._build_sorted_data, region)
        for name, hook, regions in self.update_hooks:
            if regions != "all" and region not in regions:
                continue
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(region)
                else:
                    await run_in_pool(hook, region)
                logger.info("MasterData [%s.%s] 更新后回调 [%s] 执行成功", region, self.name, name)
            except Exception:
                logger.print_exc(f"MasterData [{region}.{self.name}] 更新后回调 [{name}] 执行失败")

    async def _update_before_get(self, region: str):
        async with self.lock:
            if self.data.get(region) is None:
                try:
                    await self._load_from_cache(region)
                except Exception as e:
                    logger.warning("MasterData [%s.%s] 从本地缓存加载失败: %s", region, self.name, e)
            db_mgr = RegionMasterDbManager.get(region)
            source = await db_mgr.get_latest_source()
            if get_version_order(self.version.get(region, DEFAULT_VERSION)) < get_version_order(source.version):
                await self._download_from_db(region, source)
            if self.data.get(region) is None:
                raise Exception(f"获取 MasterData [{region}.{self.name}] 的数据失败")

    async def get_data(self, region: str):
        await self._update_before_get(region)
        return self.data[region]

    async def get_path(self, region: str):
        await self._update_before_get(region)
        return self.get_cache_path(region)

    async def get_indexed(self, region: str, key: str) -> Dict[str, Any]:
        await self._update_before_get(region)
        return self.indexed_data.get(region, {}).get(key)

    async def get_sorted(self, region: str, key: str) -> List[Any]:
        await self._update_before_get(region)
        return self.sorted_data.get(region, {}).get(key)

    @classmethod
    def get(cls, name: str) -> "MasterDataManager":
        if name not in cls._all_mgrs:
            cls._all_mgrs[name] = MasterDataManager(name)
        return cls._all_mgrs[name]

    @classmethod
    def register_updated_hook(cls, name: str, hook_name: str, hook, regions="all"):
        cls.get(name).update_hooks.append((hook_name, regions, hook))

    @classmethod
    def updated_hook(cls, name: str, hook_name: str, regions="all"):
        def _wrapper(func):
            cls.register_updated_hook(name, hook_name, func, regions)
            return func
        return _wrapper

    @classmethod
    def register_map_fn(cls, name: str, map_fn, regions="all"):
        if isinstance(regions, str):
            regions = [regions]
        for region in regions:
            cls.get(name).map_fn[region] = map_fn

    @classmethod
    def map_function(cls, name: str, regions="all"):
        def _wrapper(func):
            cls.register_map_fn(name, func, regions)
            return func
        return _wrapper

    @classmethod
    def register_download_fn(cls, name: str, download_fn, regions="all"):
        if isinstance(regions, str):
            regions = [regions]
        for region in regions:
            cls.get(name).download_fn[region] = download_fn

    @classmethod
    def download_function(cls, name: str, regions="all"):
        def _wrapper(func):
            cls.register_download_fn(name, func, regions)
            return func
        return _wrapper

    @classmethod
    def set_index_keys(cls, name: str, index_keys: Union[str, List[str], Dict[str, List[str]]]):
        cls.get(name)._set_index_keys(index_keys)

    @classmethod
    def set_sort_keys(cls, name: str, sort_keys: Union[str, List[str], Dict[str, List[str]]]):
        cls.get(name)._set_sort_keys(sort_keys)


class RegionMasterDataWrapper:
    def __init__(self, region: str, name: str):
        self.region = region
        self.mgr = MasterDataManager.get(name)

    async def get(self):
        return await self.mgr.get_data(self.region)

    async def get_path(self):
        return await self.mgr.get_path(self.region)

    async def get_indexed(self, key: str):
        return await self.mgr.get_indexed(self.region, key)

    async def get_sorted(self, key: str):
        return await self.mgr.get_sorted(self.region, key)

    async def find_by(self, key: str, value: Any, mode="first"):
        ind = await self.get_indexed(key)
        if ind is not None:
            ret = ind.get(value)
            if not ret:
                return [] if mode == "all" else None
            if mode == "first":
                return ret[0]
            if mode == "last":
                return ret[-1]
            if mode == "all":
                return ret
            raise ValueError(f"未知的查找模式: {mode}")
        data = await self.get()
        return find_by(data, key, value, mode)

    async def collect_by(self, key: str, values: Union[List[Any], Set[Any]]):
        ind = await self.get_indexed(key)
        if ind is not None:
            ret = []
            for value in values:
                if value in ind:
                    ret.extend(ind[value])
            return ret
        data = await self.get()
        values_set = set(values)
        return [item for item in data if item[key] in values_set]

    async def find_by_id(self, id: int):
        return await self.find_by("id", id)

    async def collect_by_ids(self, ids: Union[List[int], Set[int]]):
        return await self.collect_by("id", ids)


class RegionMasterDataCollection:
    def __init__(self, region: str):
        self._region = region
        self.system_live2ds = RegionMasterDataWrapper(region, "systemLive2ds")
        self.character_archive_voices = RegionMasterDataWrapper(region, "characterArchiveVoices")
        self.game_characters = RegionMasterDataWrapper(region, "gameCharacters")
        self.stamps = RegionMasterDataWrapper(region, "stamps")
        self.cards = RegionMasterDataWrapper(region, "cards")
        self.musics = RegionMasterDataWrapper(region, "musics")
        self.events = RegionMasterDataWrapper(region, "events")

    async def get(self, name: str):
        return await RegionMasterDataWrapper(self._region, name).get()

    async def get_version(self) -> str:
        mgr = RegionMasterDbManager.get(self._region)
        return (await mgr.get_latest_source()).version


MasterDataManager.set_index_keys("systemLive2ds", ["id", "characterId"])
MasterDataManager.set_index_keys("characterArchiveVoices", ["id", "gameCharacterId"])
MasterDataManager.set_index_keys("gameCharacters", ["id"])
MasterDataManager.set_index_keys("cards", ["id", "characterId"])
MasterDataManager.set_sort_keys("events", ["startAt"])


# ================================ 解包Asset资源 ================================ #

DEFAULT_RIP_ASSET_DIR = f"{SEKAI_ASSET_DIR}/rip"
DEFAULT_GET_RIP_ASSET_TIMEOUT_CFG = asset_config.item("default_rip_asset_download_timeout")
RIP_IMG_CACHE_MAX_RES_CFG = asset_config.item("rip_img_cache_max_res")

ONDEMAND_PREFIXES = ["event", "gacha", "music/long", "mysekai", "virtual_live"]
STARTAPP_PREFIXES = ["bonds_honor", "honor", "thumbnail", "character", "music", "rank_live", "stamp", "home/banner", "player_frame", "areaitem"]


def sekai_best_url_map(url: str) -> str:
    url = url.replace("_rip", "")
    if "music_score" in url:
        url = url + ".txt"
    return url


def haruki_url_map(url: str) -> str:
    idx = url.find("assets/")
    assert idx != -1, f"解包资源url格式错误: {url}"
    idx = idx + len("assets/")
    part1, part2 = url[:idx], url[idx:]
    part2 = part2.replace("_rip", "")
    if "music_score" in part2:
        part2 = part2 + ".txt"
    part2 = part2.replace(".asset", ".json")
    if any(part2.startswith(prefix) for prefix in ONDEMAND_PREFIXES):
        category = "ondemand"
    elif any(part2.startswith(prefix) for prefix in STARTAPP_PREFIXES):
        category = "startapp"
    else:
        logger.warning("在startapp和ondemand都找不到: %s", url)
        category = "ondemand"
    return f"{part1}{category}/{part2}"


def pjsekai_moe_url_map(url: str) -> str:
    return url.replace("_rip", "")


def unipjsk_url_map(url: str) -> str:
    idx = url.find("assets.unipjsk.com/")
    assert idx != -1, f"解包资源url格式错误: {url}"
    idx = idx + len("assets.unipjsk.com/")
    part1, part2 = url[:idx], url[idx:]
    part2 = part2.replace("_rip", "").replace(".asset", ".json")
    if any(part2.startswith(prefix) for prefix in ONDEMAND_PREFIXES):
        category = "ondemand"
    elif any(part2.startswith(prefix) for prefix in STARTAPP_PREFIXES):
        category = "startapp"
    else:
        logger.warning("在startapp和ondemand都找不到: %s", url)
        category = "ondemand"
    return f"{part1}{category}/{part2}"


DEFAULT_URL_MAP_METHODS = {
    "sekai.best": sekai_best_url_map,
    "haruki": haruki_url_map,
    "pjsekai.moe": pjsekai_moe_url_map,
    "unipjsk": unipjsk_url_map,
}


class RegionRipAssetSource:
    def __init__(self, name: str, base_url: str, url_map_method_name: str = None, prefixes: List[str] = None):
        self.name = name
        self.base_url = base_url
        self.url_map_method = lambda x: x
        if url_map_method_name:
            self.url_map_method = DEFAULT_URL_MAP_METHODS[url_map_method_name]
        elif self.name in DEFAULT_URL_MAP_METHODS:
            self.url_map_method = DEFAULT_URL_MAP_METHODS[self.name]
        self.prefixes = prefixes


class RegionRipAssetManger:
    _all_mgrs: dict[str, "RegionRipAssetManger"] = {}
    _img_cache_map: dict[str, Image.Image] = {}

    def __init__(self, region: str, sources: List[RegionRipAssetSource]):
        self.region = region
        self.sources = sources
        self.cache_dir = pjoin(DEFAULT_RIP_ASSET_DIR, region)
        self.cached_images: Dict[str, Image.Image] = {}
        create_folder(self.cache_dir)

    @classmethod
    def get(cls, region: str) -> "RegionRipAssetManger":
        if region not in cls._all_mgrs:
            cfg = asset_config.get_all()
            assert region in cfg and "rip" in cfg[region], f"未找到 {region} 的 RipAsset 配置"
            region_config = cfg[region]["rip"]
            cls._all_mgrs[region] = RegionRipAssetManger(
                region=region,
                sources=[RegionRipAssetSource(**source) for source in region_config["sources"]],
            )
        return cls._all_mgrs[region]

    async def _download_data(self, url: str, timeout: int) -> bytes:
        async def do_download():
            async with get_client_session().get(url, ssl=False) as resp:
                if resp.status != 200:
                    raise Exception(f"请求失败: {resp.status}")
                return await resp.read()
        if not timeout:
            return await do_download()
        return await asyncio.wait_for(do_download(), timeout)

    async def get_asset(
        self,
        path: str,
        use_cache=True,
        allow_error=True,
        default=None,
        cache_expire_secs=None,
        timeout: Union[int, ConfigItem] = DEFAULT_GET_RIP_ASSET_TIMEOUT_CFG,
    ) -> bytes:
        cache_path = pjoin(self.cache_dir, path)
        if use_cache:
            try:
                assert os.path.exists(cache_path)
                if cache_expire_secs is not None:
                    assert datetime.now().timestamp() - os.path.getmtime(cache_path) < cache_expire_secs
                with open(cache_path, "rb") as f:
                    return f.read()
            except Exception:
                pass
        error_list: List[Tuple[str, str]] = []
        for source in self.sources:
            if source.prefixes and not any(path.startswith(prefix) for prefix in source.prefixes):
                continue
            url = None
            try:
                if not source.base_url.endswith("/"):
                    source.base_url += "/"
                url = source.url_map_method(source.base_url + path)
                data = await self._download_data(url, get_cfg_or_value(timeout))
                if use_cache:
                    create_parent_folder(cache_path)
                    with open(cache_path, "wb") as f:
                        f.write(data)
                return data
            except Exception as e:
                e = get_exc_desc(e)
                error_list.append((source.name, e))
                logger.warning("从数据源 [%s] 获取 %s 解包资源 %s 失败: %s, url=%s", source.name, self.region, path, e, url)
        if not allow_error:
            error_list_text = "".join(f"[{n}] {truncate(err, 40)}\n" for n, err in error_list)
            raise Exception(f"从所有数据源获取 {self.region} 解包资源 {path} 失败:\n{error_list_text.strip()}")
        logger.warning("从所有数据源获取 %s 解包资源 %s 失败: %s，返回默认值", self.region, path, error_list)
        return default

    async def get_asset_cache_path(
        self,
        path: str,
        allow_error=True,
        default=None,
        cache_expire_secs=None,
        timeout: Union[int, ConfigItem] = DEFAULT_GET_RIP_ASSET_TIMEOUT_CFG,
    ) -> str:
        cache_path = pjoin(self.cache_dir, path)
        try:
            assert os.path.exists(cache_path)
            if cache_expire_secs is not None:
                assert datetime.now().timestamp() - os.path.getmtime(cache_path) < cache_expire_secs
            return cache_path
        except Exception:
            pass
        error_list: List[Tuple[str, str]] = []
        for source in self.sources:
            if source.prefixes and not any(path.startswith(prefix) for prefix in source.prefixes):
                continue
            url = None
            try:
                if not source.base_url.endswith("/"):
                    source.base_url += "/"
                url = source.url_map_method(source.base_url + path)
                data = await self._download_data(url, get_cfg_or_value(timeout))
                create_parent_folder(cache_path)
                with open(cache_path, "wb") as f:
                    f.write(data)
                return cache_path
            except Exception as e:
                e = get_exc_desc(e)
                error_list.append((source.name, e))
                logger.warning("从数据源 [%s] 获取 %s 解包资源 %s 失败: %s, url=%s", source.name, self.region, path, e, url)
        if not allow_error:
            error_list_text = "".join(f"[{n}] {truncate(err, 40)}\n" for n, err in error_list)
            raise Exception(f"从所有数据源获取 {self.region} 解包资源 {path} 失败:\n{error_list_text.strip()}")
        logger.warning("从所有数据源获取 %s 解包资源 %s 失败: %s，返回默认值", self.region, path, error_list)
        return default

    async def img(
        self,
        path: str,
        use_cache=True,
        allow_error=True,
        default=UNKNOWN_IMG,
        cache_expire_secs=None,
        timeout: Union[int, ConfigItem] = DEFAULT_GET_RIP_ASSET_TIMEOUT_CFG,
        use_img_cache: bool = False,
        img_cache_max_res: Union[int, ConfigItem] = RIP_IMG_CACHE_MAX_RES_CFG,
    ) -> Image.Image:
        if use_img_cache and path in self.cached_images:
            return self.cached_images[path]
        data = await self.get_asset(path, use_cache, allow_error, default, cache_expire_secs, get_cfg_or_value(timeout))
        try:
            img = open_image(io.BytesIO(data))
            if use_img_cache:
                if img_cache_max_res:
                    max_res = parse_cfg_num(get_cfg_or_value(img_cache_max_res))
                    w, h = img.size
                    if w * h > max_res:
                        scale = math.sqrt(max_res / (w * h))
                        img = resize_keep_ratio(img, scale, mode="scale")
                img_hash = get_image_pixel_hash(img)
                if img_hash in self._img_cache_map:
                    img = self._img_cache_map[img_hash]
                else:
                    self._img_cache_map[img_hash] = img
                self.cached_images[path] = img
            return img
        except Exception:
            pass
        if not allow_error:
            raise Exception(f"解析下载的 {self.region} 解包资源 {path} 为图片失败")
        logger.warning("解析下载的 %s 解包资源 %s 为图片失败: 返回默认值", self.region, path)
        return default

    async def json(
        self,
        path: str,
        use_cache=True,
        allow_error=True,
        default=None,
        cache_expire_secs=None,
        timeout: Union[int, ConfigItem] = DEFAULT_GET_RIP_ASSET_TIMEOUT_CFG,
    ) -> Any:
        data = await self.get_asset(path, use_cache, allow_error, default, cache_expire_secs, get_cfg_or_value(timeout))
        try:
            return loads_json(data)
        except Exception:
            pass
        if not allow_error:
            raise Exception(f"解析下载的 {self.region} 解包资源 {path} 为json失败")
        logger.warning("解析下载的 %s 解包资源 %s 为json失败: 返回默认值", self.region, path)
        return default


def live2d_voice_path(bundle: str, voice: str) -> str:
    return f"sound/system_live2d/voice/{bundle}/{voice}.mp3"


def character_system_voice_folder(character_id: int, given_name_en: str) -> str:
    return f"{int(character_id):02d}{str(given_name_en or '').strip().lower()}"


def archive_voice_path(character_id: int, given_name_en: str, asset_name: str) -> str:
    folder = character_system_voice_folder(character_id, given_name_en)
    return f"sound/system/voice/{folder}/{asset_name}.mp3"