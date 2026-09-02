from openai import AsyncOpenAI

from ..api_provider import ApiProvider


class YunWuApiProvider(ApiProvider):
    def __init__(self):
        super().__init__(name="yunwu", code="yw")

    def get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_url(),
        )

    async def sync_quota(self):
        return None