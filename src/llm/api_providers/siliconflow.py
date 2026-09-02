from openai import AsyncOpenAI

from src.utils import *

from ..api_provider import ApiProvider


class SiliconflowApiProvider(ApiProvider):
    def __init__(self):
        super().__init__(name="siliconflow", code="sf")

    def get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_url(),
        )
        
    async def sync_quota(self):
        url = self.config.get("user_info_url")
        headers = {"Authorization": f"Bearer {self.get_api_key()}"}
        async with get_client_session().get(url, headers=headers) as resp:
            if resp.status != 200:
                raise Exception(f"获取SiliconFlow剩余额度失败: {resp.status} {resp.reason}")
            data = await resp.json()
            return float(data['data']['totalBalance'])

