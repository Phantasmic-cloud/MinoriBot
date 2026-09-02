from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from openai import AsyncOpenAI

from src.utils import *

logger = get_logger("llm")
file_db = get_file_db("data/llm/db.json")
utils_logger = logger


@dataclass
class LlmModel:
    name: str
    input_pricing: float = 0.0
    output_pricing: float = 0.0
    max_token: int = 128000
    is_multimodal: bool = False
    model_id: Optional[str] = None
    image_response: bool = False
    allow_online: bool = False
    provider: "ApiProvider" = None
    data: dict = field(default_factory=dict)
    client_kwargs: dict = field(default_factory=dict)
    extra_body: dict = field(default_factory=dict)

    def calc_price(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.input_pricing + output_tokens * self.output_pricing

    def get_model_id(self) -> str:
        return self.model_id or self.name

    def get_price_unit(self) -> str:
        return self.provider.get_price_unit()

    def get_full_name(self) -> str:
        return f"{self.provider.name}:{self.name}"


class ApiProvider:
    def __init__(self, name: str, code: str):
        self.name = name
        self.code = code
        self.config = Config(f"llm.providers.{name}")
        self.models: List[LlmModel] = []
        self.models_mtime = None
        self.cur_query_ts = 0
        self.cur_sec_query_count = 0
        self.local_quota_key = f"api_provider_{name}_local_quota"
        self.last_quota_sync_time = datetime.now()

    def get_qps_limit(self) -> int:
        return self.config.get("qps_limit")

    def get_quota_sync_interval_sec(self) -> int:
        return parse_cfg_num(self.config.get("quota_sync_interval_sec"))

    def get_price_unit(self) -> str:
        return self.config.get("price_unit")

    def get_api_key(self) -> str:
        return self.config.get("api_key")

    def get_base_url(self) -> str:
        return self.config.get("base_url")

    def update_models(self):
        mtime = self.config.mtime()
        if self.models_mtime == mtime:
            return

        def parse_price(d, k):
            if not isinstance(d.get(k), str):
                return
            nums = d[k].split("/", 1) if "/" in d[k] else [d[k], "1"]
            d[k] = float(nums[0]) / float(nums[1])

        self.models = []
        for model_config in self.config.get("models", []):
            parse_price(model_config, "input_pricing")
            parse_price(model_config, "output_pricing")
            self.models.append(LlmModel(**model_config))
        for model in self.models:
            model.provider = self
        self.models_mtime = mtime
        logger.info("API供应方 %s 模型列表更新成功 (共 %s 个模型)", self.name, len(self.models))

    def check_qps_limit(self):
        now_ts = int(datetime.now().timestamp())
        if now_ts > self.cur_query_ts:
            self.cur_query_ts = now_ts
            self.cur_sec_query_count = 0
        qps_limit = self.get_qps_limit()
        if self.cur_sec_query_count >= qps_limit:
            logger.warning("API供应方 %s QPS限制 %s 已超出", self.name, qps_limit)
            raise Exception(f"API供应方 {self.name} QPS限制 {qps_limit} 已超出，请稍后再试")
        self.cur_sec_query_count += 1

    async def aupdate_quota(self, delta: float) -> float:
        local_quota = file_db.get(self.local_quota_key, 0.0)
        if not isinstance(local_quota, (int, float)):
            local_quota = 0.0
        last_quota = local_quota
        local_quota += delta
        file_db.set(self.local_quota_key, local_quota)
        new_quota = await self.aget_current_quota()
        price_unit = self.get_price_unit()
        logger.info(
            "API供应方 %s 更新剩余额度成功: %s%s -> %s%s",
            self.name, last_quota, price_unit, new_quota, price_unit,
        )
        return new_quota

    async def aget_current_quota(self) -> float:
        if (datetime.now() - self.last_quota_sync_time).total_seconds() > self.get_quota_sync_interval_sec():
            try:
                new_quota = await self.sync_quota()
                if new_quota is not None:
                    file_db.set(self.local_quota_key, new_quota)
                    logger.info("API供应方 %s 同步剩余额度成功: %s%s", self.name, new_quota, self.get_price_unit())
            except Exception:
                logger.print_exc(f"API供应方 {self.name} 同步剩余额度失败")
            self.last_quota_sync_time = datetime.now()
        return file_db.get(self.local_quota_key, 0.0)

    def get_client(self) -> AsyncOpenAI:
        raise NotImplementedError()

    async def sync_quota(self):
        return None