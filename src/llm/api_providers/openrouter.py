from openai import AsyncOpenAI

from src.utils import *

from ..api_provider import ApiProvider


class OpenrouterApiProvider(ApiProvider):
    def __init__(self):
        super().__init__(name="openrouter", code="or")

    def get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_url(),
        )
        
    async def sync_quota(self):
        url = self.config.get('auth_key_url')
        headers = {"Authorization": f"Bearer {self.get_api_key()}"}
        async with get_client_session().get(url, headers=headers) as resp:
            if resp.status != 200:
                raise Exception(f"获取OpenRouter剩余额度失败: {resp.status} {resp.reason}")
            data = await resp.json()
            usage, limit = data['data']["usage"], data['data']["limit"]
            if limit is None:
                raise Exception("OpenRouter API key 无限额")
            return limit - usage

