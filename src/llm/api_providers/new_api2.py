from openai import AsyncOpenAI

from ..api_provider import ApiProvider


class NewApi2ApiProvider(ApiProvider):
    def __init__(self):
        super().__init__(name="new-api2", code="na2")

    def get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_url(),
        )

    async def sync_quota(self):
        return None



