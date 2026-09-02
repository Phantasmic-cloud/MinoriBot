from typing import Any


class MinoriError(Exception):
    """框架基类异常。"""


class ActionFailed(MinoriError):
    """OneBot API 返回失败（retcode != 0 或 status != ok）。"""

    def __init__(
        self,
        action: str,
        retcode: int = -1,
        message: str = "",
        wording: str = "",
        data: Any = None,
    ) -> None:
        self.action = action
        self.retcode = retcode
        self.message = message
        self.wording = wording
        self.data = data
        detail = wording or message or f"retcode={retcode}"
        super().__init__(f"{action} 失败: {detail}")


class ApiTimeout(MinoriError):
    """等待 API 回包超时。"""

    def __init__(self, action: str, timeout: float) -> None:
        self.action = action
        self.timeout = timeout
        super().__init__(f"{action} 超时 ({timeout}s)")


class StopPropagation(Exception):
    """handler 里抛出后，后续 handler 不再执行。"""


class ReplyException(MinoriError):
    """业务失败，适合直接回给用户看。"""


class NoReplyException(Exception):
    """退出当前指令处理，不回消息。"""
