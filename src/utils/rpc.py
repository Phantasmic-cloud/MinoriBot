import asyncio
from typing import Callable

import aiorpcx

from src.core import ConfigItem, get_cfg_or_value
from src.utils.utils import async_task

_rpc_service_tokens: dict[str, str | ConfigItem] = {}
_rpc_handlers: dict[str, Callable] = {}


def rpc_method(service_name: str, method_name: str):
    """装饰器，用于注册 RPC 方法处理程序。"""

    def decorator(func):
        _rpc_handlers[service_name + "." + method_name] = func
        return func

    return decorator


class RpcSession(aiorpcx.RPCSession):
    def __init__(
        self,
        name: str,
        logger,
        *args,
        on_connect: Callable | None = None,
        on_disconnect: Callable | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.id = str(self.remote_address())
        self.name = name
        self._logger = logger
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.processing_timeout = 300.0
        self.sent_request_timeout = 60.0
        if self.on_connect:
            self.on_connect(self)
        logger.info("%sRPC服务的客户端 %s 连接成功", self.name, self.id)

    async def connection_lost(self):
        await super().connection_lost()
        if self.on_disconnect:
            self.on_disconnect(self)
        self._logger.info("%sRPC服务的客户端 %s 断开连接", self.name, self.id)

    async def handle_request(self, request):
        self._logger.debug("收到%sRPC服务的客户端 %s 的请求 %s", self.name, self.id, request.method)
        handler_fn = _rpc_handlers.get(self.name + "." + request.method)
        token = get_cfg_or_value(_rpc_service_tokens[self.name])

        if not request.args or request.args[0] != token:
            self._logger.warning("%sRPC服务的客户端 %s 提供了无效或缺失的令牌", self.name, self.id)
            await asyncio.sleep(1.0)
            raise aiorpcx.RPCError(-32000, "Invalid or missing token")

        if handler_fn is None:
            raise aiorpcx.RPCError(-32601, f"Unknown method {request.method}")

        args = request.args[1:]
        request.args = [self.id] + args
        resp = await aiorpcx.handler_invocation(handler_fn, request)()
        self._logger.debug(
            "%sRPC服务的客户端 %s 的请求 %s %s 返回: %s",
            self.name, self.id, request.method, args, resp,
        )
        return resp


def get_session_factory(
    name: str,
    logger,
    on_connect: Callable | None = None,
    on_disconnect: Callable | None = None,
):
    def factory(*args, **kwargs):
        return RpcSession(
            name,
            logger,
            *args,
            on_connect=on_connect,
            on_disconnect=on_disconnect,
            **kwargs,
        )

    return factory


def start_rpc_service(
    host: str,
    port: int,
    name: str,
    token: str | ConfigItem,
    logger,
    on_connect: Callable | None = None,
    on_disconnect: Callable | None = None,
):
    """
    启动 RPC 服务。
    token 作为客户端请求的第一个参数校验。
    """
    _rpc_service_tokens[name] = token

    @async_task(f"{name}RPC服务", logger)
    async def _run():
        try:
            async with aiorpcx.serve_ws(
                get_session_factory(name, logger, on_connect, on_disconnect),
                host,
                port,
            ):
                logger.info("%sRPC服务已启动 ws://%s:%s", name, host, port)
                await asyncio.sleep(1e9)
        except asyncio.CancelledError:
            logger.info("%sRPC服务已关闭", name)
