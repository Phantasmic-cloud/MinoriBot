import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Union

from src.utils import *

from .api_provider import ApiProvider, LlmModel
from .api_provider_manager import api_provider_mgr

__all__ = [
    "ApiProvider",
    "ChatSession",
    "ChatSessionResponse",
    "LlmModel",
    "api_provider_mgr",
    "get_model_preset",
    "get_text_embedding",
    "translate_text",
    "tts",
]

config = Config("llm.llm")
logger = get_logger("llm")
file_db = get_file_db("data/llm/db.json")

CHAT_TIMEOUT_CFG = config.item("chat_timeout")
CHAT_MODEL_SWITCH_INTERVAL_CFG = config.item("chat_model_switch_interval")
CHAT_MAX_TOKENS_CFG = config.item("chat_max_tokens")
session_id_top = 0


def get_model_preset(key: str) -> Union[str, List[str], dict]:
    """从 llm.model_preset 里取模型预设，`&xxx` 会再解一层引用。"""
    ret = Config("llm.model_preset").get(key)

    def parse_ref(s: str):
        return get_model_preset(s[1:]) if s.startswith("&") else s

    if isinstance(ret, str):
        return parse_ref(ret)
    if isinstance(ret, list):
        return [parse_ref(s) for s in ret]
    if isinstance(ret, dict):
        return {k: parse_ref(v) for k, v in ret.items()}
    return ret


@dataclass
class ChatSessionResponse:
    result: str
    provider: ApiProvider
    model: LlmModel
    prompt_tokens: int
    completion_tokens: int
    cost: float
    quota: float
    reasoning: Optional[str] = None
    images: list = field(default_factory=list)
    result_list: list = field(default_factory=list)


class ChatSession:
    """一次聊天会话：拼消息、选模型、拿回复。"""
    @staticmethod
    def check_model_name(model_name: Union[str, List[str]], mode="text"):
        if not isinstance(model_name, str):
            for name in model_name:
                ChatSession.check_model_name(name, mode=mode)
            return
        model = api_provider_mgr.find_model(model_name)
        if mode == "mm" and not model.is_multimodal:
            raise Exception(f"模型 {model_name} 不支持多模态输入")
        if mode == "image" and not model.image_response:
            raise Exception(f"模型 {model_name} 不支持图片回复")

    def __init__(self, system_prompt=None):
        global session_id_top
        session_id_top += 1
        self.id = session_id_top
        logger.info("创建会话%s", self.id)
        self.content = []
        self.has_image = False
        if system_prompt:
            self.append_system_content(system_prompt, verbose=False)
        self.update_time = datetime.now()

    def append_content(self, role, text, imgs=None, verbose=True):
        if not text and not imgs:
            logger.warning("会话%s跳过添加空消息", self.id)
            return
        if imgs is None:
            imgs = []
        for i in range(len(imgs)):
            try:
                from PIL import Image
                if isinstance(imgs[i], Image.Image):
                    imgs[i] = get_image_b64(imgs[i])
            except Exception:
                pass
        if imgs:
            content = [{"type": "text", "text": text}]
            for img in imgs:
                content.append({"type": "image_url", "image_url": {"url": img}})
            self.has_image = True
        else:
            content = text
        self.content.append({"role": role, "content": content})
        if verbose:
            log_text = f"会话{self.id}添加{role}_content: \"{str(text).replace(chr(10), '\\n')}\""
            if imgs:
                log_text += f" + {len(imgs)}img(s)"
            log_text += f", 目前会话长度:{len(self)}"
            logger.info(log_text)
        self.update_time = datetime.now()

    def append_system_content(self, text, verbose=True):
        self.append_content("system", text, verbose=verbose)

    def append_user_content(self, text, imgs=None, verbose=True):
        self.append_content("user", text, imgs, verbose=verbose)

    def append_bot_content(self, text, imgs=None, verbose=True):
        self.append_content("assistant", text, imgs, verbose=verbose)

    def limit_length(self, limit: int, drop="oldest"):
        system_content = None
        if self.content and self.content[0]["role"] == "system":
            system_content = self.content[0]
            self.content = self.content[1:]
        if len(self.content) >= limit:
            if drop == "oldest":
                self.content = self.content[-limit:]
            else:
                self.content = self.content[:limit]
        if system_content:
            self.content.insert(0, system_content)

    def __len__(self):
        return len(self.content)

    def clear_content(self):
        self.content = []
        self.has_image = False
        self.update_time = datetime.now()

    def has_multimodal_content(self):
        return self.has_image

    async def get_response(
        self,
        model_name: Union[str, List[str]],
        process_func=None,
        image_response=False,
        timeout: Union[int, ConfigItem] = CHAT_TIMEOUT_CFG,
        model_switch_interval: Union[int, ConfigItem] = CHAT_MODEL_SWITCH_INTERVAL_CFG,
        max_tokens: Union[int, ConfigItem] = CHAT_MAX_TOKENS_CFG,
    ):
        if isinstance(model_name, str):
            model_name = [model_name]
        errs: List[Tuple[str, str]] = []
        for idx, name in enumerate(model_name):
            try:
                model = api_provider_mgr.find_model(name)
                name = model.get_full_name()
                provider = model.provider
                if not model.is_multimodal and self.has_image:
                    raise Exception(f"模型 {name} 不支持多模态输入")
                logger.info("会话%s请求回复, 模型名: %s (%s/%s)", self.id, name, idx + 1, len(model_name))
                provider.check_qps_limit()
                extra_body = model.extra_body.copy()
                if model.image_response:
                    extra_body["image_response"] = image_response
                    extra_body["modalities"] = ["image", "text"]
                client = provider.get_client()
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model.get_model_id(),
                            messages=self.content,
                            extra_body=extra_body,
                            max_tokens=get_cfg_or_value(max_tokens),
                            **model.client_kwargs,
                        ),
                        timeout=get_cfg_or_value(timeout),
                    )
                except TimeoutError:
                    raise Exception("等待回复超时")
                if not isinstance(response, dict):
                    response = response.model_dump()
                if response.get("error"):
                    raise Exception(response["error"])
                message = response["choices"][0]["message"]
                prompt_tokens = response["usage"]["prompt_tokens"]
                completion_tokens = response["usage"]["completion_tokens"]
                resp_content = message["content"]
                result = ""
                images = []
                if isinstance(resp_content, str):
                    result = resp_content
                    result_list = [result]
                else:
                    result_list = []
                    for part in resp_content:
                        if isinstance(part, str):
                            result += part
                            result_list.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            result += part["text"]
                            result_list.append(part["text"])
                        elif isinstance(part, dict) and part.get("type") == "image_url":
                            img = b64_to_image(part["image_url"]["url"])
                            images.append(img)
                            result_list.append(img)
                        else:
                            images.append(part)
                            result_list.append(part)
                for item in message.get("images", []):
                    img = b64_to_image(item["image_url"]["url"])
                    images.append(img)
                    result_list.append(img)
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                log_text = f"会话{self.id}获取回复，使用token: {prompt_tokens}+{completion_tokens}，内容:\n"
                if reasoning:
                    log_text += "【思考】" + truncate(reasoning.replace("\n", "\\n"), 128) + "\n"
                for part in result_list:
                    log_text += truncate(part.replace("\n", "\\n"), 128) if isinstance(part, str) else "[图片]"
                logger.info(log_text)
                self.append_bot_content(result, imgs=[get_image_b64(img) for img in images], verbose=False)
                cost = model.calc_price(prompt_tokens, completion_tokens)
                quota = await provider.aupdate_quota(-cost)
                self.update_time = datetime.now()
                ret = ChatSessionResponse(
                    result=result,
                    provider=provider,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=cost,
                    quota=quota,
                    reasoning=reasoning,
                    images=images,
                    result_list=result_list,
                )
                if process_func:
                    ret = await process_func(ret) if asyncio.iscoroutinefunction(process_func) else process_func(ret)
                return ret
            except Exception as e:
                logger.print_exc(f"会话{self.id}获取回复失败, 使用模型 {name}: {get_exc_desc(e)}")
                errs.append((name, get_exc_desc(e)))
                await asyncio.sleep(get_cfg_or_value(model_switch_interval))
        if len(errs) == 1:
            raise ReplyException(f"调用模型{errs[0][0]}失败:\n{truncate(errs[0][1], 64)}")
        err_str = "\n".join(f"[{err[0]}] {err[1]}" for err in errs)
        raise ReplyException(f"调用多个模型失败:\n{truncate(err_str, 64)}")


async def get_text_embedding(texts: List[str], model_name: str) -> List[List[float]]:
    """批量拿文本嵌入向量。"""
    logger.info("获取文本嵌入: %s", texts)
    models = config.get("text_embedding_models")
    model = find_by(models, "name", model_name)
    assert model is not None, f"文本嵌入模型 {model_name} 不存在"
    provider = api_provider_mgr.get_provider(model["provider"])
    assert provider is not None, f"文本嵌入模型 {model_name} 的供应方 {model['provider']} 不存在"
    response = await provider.get_client().embeddings.create(
        input=texts,
        model=model["id"],
        encoding_format="float",
    )
    embeddings = [d.embedding for d in response.data]
    tokens = response.usage.prompt_tokens
    cost = model["input_pricing"] * tokens
    await provider.aupdate_quota(-cost)
    return embeddings


async def tts(text, save_path: str):
    """把文本合成语音写到文件。"""
    logger.info("TTS: %s", text)
    models = config.get("tts_models")
    assert models, "TTS模型列表为空"
    model = models[0]
    provider = api_provider_mgr.get_provider(model["provider"])
    provider.check_qps_limit()
    response = await provider.get_client().audio.speech.create(
        model=model["id"],
        voice=model["voice"],
        input=text,
    )
    response.write_to_file(save_path)
    logger.info("TTS成功, 保存到: %s", save_path)
    return save_path


async def translate_text(
    text,
    additional_info=None,
    dst_lang="中文",
    timeout=20,
    default=None,
    model=None,
    cache=True,
):
    """把文本翻到目标语言，可按 md5 缓存。"""
    if model is None:
        model = get_model_preset("translation")
    text_translation_db = get_file_db("data/llm/text_translations.json")
    translations = text_translation_db.get("translations", {}) if cache else {}
    key = get_md5(text)
    if not cache or key not in translations:
        logger.info(
            "翻译文本: %s 额外信息: %s 目标语言: %s",
            truncate(text, 64),
            truncate(additional_info, 64),
            dst_lang,
        )
        try:
            session = ChatSession()
            extra = f"额外的参考信息:\"{additional_info}\"，" if additional_info else ""
            prompt = f"翻译文本到{dst_lang}{extra}，请直接输出翻译结果并结束，不要包含其他内容:\n{text}"
            session.append_user_content(prompt)
            response = await asyncio.wait_for(session.get_response(model), timeout=timeout)
            result = response.result.strip()
            logger.info("翻译结果: %s", truncate(result, 64))
            translations[key] = result
        except Exception as e:
            logger.print_exc(f"翻译失败: {e}")
            return default
    if cache:
        text_translation_db.set("translations", translations)
    return translations[key]