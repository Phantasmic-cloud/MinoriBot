from typing import Any

from .message import MessageLike, dump_message


class APIMixin:
    """OneBot v11 常用 API。插件通过 bot.xxx(...) 调用，底层都走 call_api。"""

    async def call_api(self, action: str, **params: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    # -------- 消息 -------- #

    async def send_private_msg(
        self,
        user_id: int,
        message: MessageLike,
        auto_escape: bool = False,
        **extra: Any,
    ) -> Any:
        return await self.call_api(
            "send_private_msg",
            user_id=int(user_id),
            message=dump_message(message, auto_escape),
            auto_escape=auto_escape,
            **extra,
        )

    async def send_group_msg(
        self,
        group_id: int,
        message: MessageLike,
        auto_escape: bool = False,
        **extra: Any,
    ) -> Any:
        return await self.call_api(
            "send_group_msg",
            group_id=int(group_id),
            message=dump_message(message, auto_escape),
            auto_escape=auto_escape,
            **extra,
        )

    async def send_msg(
        self,
        message: MessageLike,
        message_type: str | None = None,
        user_id: int | None = None,
        group_id: int | None = None,
        auto_escape: bool = False,
        **extra: Any,
    ) -> Any:
        params: dict[str, Any] = {
            "message": dump_message(message, auto_escape),
            "auto_escape": auto_escape,
            **extra,
        }
        if message_type:
            params["message_type"] = message_type
        if user_id is not None:
            params["user_id"] = int(user_id)
        if group_id is not None:
            params["group_id"] = int(group_id)
        return await self.call_api("send_msg", **params)

    async def delete_msg(self, message_id: int, **extra: Any) -> Any:
        return await self.call_api("delete_msg", message_id=int(message_id), **extra)

    async def get_msg(self, message_id: int, **extra: Any) -> Any:
        return await self.call_api("get_msg", message_id=int(message_id), **extra)

    async def get_forward_msg(self, id: str, **extra: Any) -> Any:
        return await self.call_api("get_forward_msg", id=str(id), **extra)

    async def send_like(self, user_id: int, times: int = 1, **extra: Any) -> Any:
        return await self.call_api("send_like", user_id=int(user_id), times=int(times), **extra)

    async def send_group_forward_msg(self, group_id: int, messages: list[Any], **extra: Any) -> Any:
        return await self.call_api(
            "send_group_forward_msg",
            group_id=int(group_id),
            messages=messages,
            **extra,
        )

    async def send_private_forward_msg(self, user_id: int, messages: list[Any], **extra: Any) -> Any:
        return await self.call_api(
            "send_private_forward_msg",
            user_id=int(user_id),
            messages=messages,
            **extra,
        )

    # -------- 群管理 -------- #

    async def set_group_kick(
        self,
        group_id: int,
        user_id: int,
        reject_add_request: bool = False,
        **extra: Any,
    ) -> Any:
        return await self.call_api(
            "set_group_kick",
            group_id=int(group_id),
            user_id=int(user_id),
            reject_add_request=reject_add_request,
            **extra,
        )

    async def set_group_ban(
        self,
        group_id: int,
        user_id: int,
        duration: int = 1800,
        **extra: Any,
    ) -> Any:
        return await self.call_api(
            "set_group_ban",
            group_id=int(group_id),
            user_id=int(user_id),
            duration=int(duration),
            **extra,
        )

    async def set_group_anonymous_ban(
        self,
        group_id: int,
        anonymous: dict[str, Any] | None = None,
        flag: str | None = None,
        duration: int = 1800,
        **extra: Any,
    ) -> Any:
        params: dict[str, Any] = {
            "group_id": int(group_id),
            "duration": int(duration),
            **extra,
        }
        if anonymous is not None:
            params["anonymous"] = anonymous
        if flag is not None:
            params["anonymous_flag"] = flag
            params["flag"] = flag
        return await self.call_api("set_group_anonymous_ban", **params)

    async def set_group_whole_ban(self, group_id: int, enable: bool = True, **extra: Any) -> Any:
        return await self.call_api(
            "set_group_whole_ban",
            group_id=int(group_id),
            enable=enable,
            **extra,
        )

    async def set_group_admin(
        self,
        group_id: int,
        user_id: int,
        enable: bool = True,
        **extra: Any,
    ) -> Any:
        return await self.call_api(
            "set_group_admin",
            group_id=int(group_id),
            user_id=int(user_id),
            enable=enable,
            **extra,
        )

    async def set_group_anonymous(self, group_id: int, enable: bool = True, **extra: Any) -> Any:
        return await self.call_api(
            "set_group_anonymous",
            group_id=int(group_id),
            enable=enable,
            **extra,
        )

    async def set_group_card(
        self,
        group_id: int,
        user_id: int,
        card: str = "",
        **extra: Any,
    ) -> Any:
        return await self.call_api(
            "set_group_card",
            group_id=int(group_id),
            user_id=int(user_id),
            card=str(card),
            **extra,
        )

    async def set_group_name(self, group_id: int, group_name: str, **extra: Any) -> Any:
        return await self.call_api(
            "set_group_name",
            group_id=int(group_id),
            group_name=str(group_name),
            **extra,
        )

    async def set_group_leave(self, group_id: int, is_dismiss: bool = False, **extra: Any) -> Any:
        return await self.call_api(
            "set_group_leave",
            group_id=int(group_id),
            is_dismiss=is_dismiss,
            **extra,
        )

    async def set_group_special_title(
        self,
        group_id: int,
        user_id: int,
        special_title: str = "",
        duration: int = -1,
        **extra: Any,
    ) -> Any:
        return await self.call_api(
            "set_group_special_title",
            group_id=int(group_id),
            user_id=int(user_id),
            special_title=str(special_title),
            duration=int(duration),
            **extra,
        )

    # -------- 请求处理 -------- #

    async def set_friend_add_request(
        self,
        flag: str,
        approve: bool = True,
        remark: str = "",
        **extra: Any,
    ) -> Any:
        return await self.call_api(
            "set_friend_add_request",
            flag=str(flag),
            approve=approve,
            remark=str(remark),
            **extra,
        )

    async def set_group_add_request(
        self,
        flag: str,
        sub_type: str,
        approve: bool = True,
        reason: str = "",
        **extra: Any,
    ) -> Any:
        return await self.call_api(
            "set_group_add_request",
            flag=str(flag),
            sub_type=str(sub_type),
            type=str(sub_type),
            approve=approve,
            reason=str(reason),
            **extra,
        )

    # -------- 账号 / 信息 -------- #

    async def get_login_info(self, **extra: Any) -> Any:
        return await self.call_api("get_login_info", **extra)

    async def get_stranger_info(self, user_id: int, no_cache: bool = False, **extra: Any) -> Any:
        return await self.call_api(
            "get_stranger_info",
            user_id=int(user_id),
            no_cache=no_cache,
            **extra,
        )

    async def get_friend_list(self, **extra: Any) -> Any:
        return await self.call_api("get_friend_list", **extra)

    async def get_group_info(self, group_id: int, no_cache: bool = False, **extra: Any) -> Any:
        return await self.call_api(
            "get_group_info",
            group_id=int(group_id),
            no_cache=no_cache,
            **extra,
        )

    async def get_group_list(self, **extra: Any) -> Any:
        return await self.call_api("get_group_list", **extra)

    async def get_group_member_info(
        self,
        group_id: int,
        user_id: int,
        no_cache: bool = False,
        **extra: Any,
    ) -> Any:
        return await self.call_api(
            "get_group_member_info",
            group_id=int(group_id),
            user_id=int(user_id),
            no_cache=no_cache,
            **extra,
        )

    async def get_group_member_list(self, group_id: int, **extra: Any) -> Any:
        return await self.call_api("get_group_member_list", group_id=int(group_id), **extra)

    async def get_group_honor_info(self, group_id: int, type: str = "all", **extra: Any) -> Any:
        return await self.call_api(
            "get_group_honor_info",
            group_id=int(group_id),
            type=str(type),
            **extra,
        )

    # -------- 媒体 / 状态 -------- #

    async def get_cookies(self, domain: str = "", **extra: Any) -> Any:
        return await self.call_api("get_cookies", domain=str(domain), **extra)

    async def get_csrf_token(self, **extra: Any) -> Any:
        return await self.call_api("get_csrf_token", **extra)

    async def get_credentials(self, domain: str = "", **extra: Any) -> Any:
        return await self.call_api("get_credentials", domain=str(domain), **extra)

    async def get_record(self, file: str, out_format: str = "mp3", **extra: Any) -> Any:
        return await self.call_api("get_record", file=str(file), out_format=str(out_format), **extra)

    async def get_image(self, file: str, **extra: Any) -> Any:
        return await self.call_api("get_image", file=str(file), **extra)

    async def can_send_image(self, **extra: Any) -> Any:
        return await self.call_api("can_send_image", **extra)

    async def can_send_record(self, **extra: Any) -> Any:
        return await self.call_api("can_send_record", **extra)

    async def get_status(self, **extra: Any) -> Any:
        return await self.call_api("get_status", **extra)

    async def get_version_info(self, **extra: Any) -> Any:
        return await self.call_api("get_version_info", **extra)

    async def set_restart(self, delay: int = 0, **extra: Any) -> Any:
        return await self.call_api("set_restart", delay=int(delay), **extra)

    async def clean_cache(self, **extra: Any) -> Any:
        return await self.call_api("clean_cache", **extra)