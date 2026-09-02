from openai import AsyncOpenAI

from ..api_provider import ApiProvider


class AiyydsApiProvider(ApiProvider):
    def __init__(self):
        super().__init__(name="ai-yyds", code="ay")

    def get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.get_api_key(),
            base_url=self.get_base_url(),
        )



