from openai import AsyncOpenAI

from src.utils import *

from ..api_provider import ApiProvider


class DeepseekApiProvider(ApiProvider):
    def __init__(self):
        super().__init__(name="deepseek", code="ds")

    def get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_url(),
        )

    async def sync_quota(self):
        url = "https://api.deepseek.com/user/balance"
        headers = {"Authorization": f"Bearer {self.get_api_key()}"}
        async with get_client_session().get(url, headers=headers) as resp:
            if resp.status != 200:
                raise Exception(f"获取DeepSeek余额失败: {resp.status}")
            data = await resp.json()
            total = sum(float(b['total_balance']) for b in data.get('balance_infos', []))
            return total