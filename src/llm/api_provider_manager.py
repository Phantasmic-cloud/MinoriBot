from typing import List, Optional, Tuple

from src.utils import *

from .api_provider import ApiProvider, LlmModel
from .api_providers.aiyyds import AiyydsApiProvider
from .api_providers.openrouter import OpenrouterApiProvider
from .api_providers.siliconflow import SiliconflowApiProvider
from .api_providers.google import GoogleApiProvider
from .api_providers.new_api import NewApiApiProvider
from .api_providers.new_api2 import NewApi2ApiProvider
from .api_providers.deepseek import DeepseekApiProvider
from .api_providers.yunwu import YunWuApiProvider
from .api_providers.longcat import LongcatApiProvider


class ApiProviderManager:
    """
    管理所有供应方和供应方的模型
    """

    def __init__(self, providers: List[AiyydsApiProvider]):
        self.providers = providers

    def get_provider(self, name_or_code: str) -> Optional[ApiProvider]:
        """
        根据名称或代号获取供应方
        """
        for provider in self.providers:
            if provider.code == name_or_code or provider.name == name_or_code:
                provider.update_models()
                return provider
        return None

    def get_all_providers(self) -> List[ApiProvider]:
        """
        获取所有供应方
        """
        for provider in self.providers:
            provider.update_models()
        return self.providers

    def get_all_models(self) -> List[LlmModel]:
        """
        获取所有model
        """
        ret = []
        for provider in self.providers:
            provider.update_models()
            for model in provider.models:
                ret.append(model)
        return ret

    def get_provider_models(self, name_or_code: str) -> Optional[List[LlmModel]]:
        """
        获取供应方的所有模型
        """
        provider = self.get_provider(name_or_code)
        if provider:
            provider.update_models()
            return provider.models
        return None

    def get_closest_provider(self, name_or_code: str) -> Optional[ApiProvider]:
        """
        获取最接近的供应方
        """
        min_distance = 1e10
        closest_provider = None
        for provider in self.providers:
            distance = min(
                levenshtein_distance(provider.name, name_or_code), 
                levenshtein_distance(provider.code, name_or_code)
            )
            if distance < min_distance:
                min_distance = distance
                closest_provider = provider
        return closest_provider

    def get_closest_model(self, name: str, provider: ApiProvider=None) -> Optional[LlmModel]:
        """
        获取最接近的模型
        """
        providers = [provider] if provider else self.providers
        min_distance = 1e10
        closest_model = None
        for provider in providers:
            provider.update_models()
            for model in provider.models:
                distance = levenshtein_distance(model.name, name)
                if distance < min_distance:
                    min_distance = distance
                    closest_model = model
        return closest_model

    def split_provider_model_name(self, name: str) -> Tuple[Optional[str], str]:
        """
        分离供应商名和模型名
        """
        provider_name = None
        model_name = name.replace("：", ":")
        if ':' in model_name:
            provider_name, model_name = model_name.split(':', maxsplit=1)
        return provider_name, model_name

    def find_model(self, model_name: str) -> LlmModel:
        """
        根据模型名称获取模型
        """
        provider_name, model_name = self.split_provider_model_name(model_name)
        
        # 如果没有指定供应方，在所有里搜索
        if not provider_name:
            res: List[Tuple[ApiProvider, LlmModel]] = []
            for provider in self.providers:
                for model in provider.models:
                    if model.name == model_name:
                        res.append((provider, model))
            # 唯一对应，直接返回
            if len(res) == 1:
                return res[0][1]
            # 找不到模型
            if len(res) == 0:
                model = self.get_closest_model(model_name)
                msg = f"未找到模型 {model_name}"
                if model:
                    msg += f"，是否是 {model.get_full_name()} ?"
                raise Exception(msg)
            # 多个供应方有同名模型，提示
            raise Exception(f"存在多个同名模型: {', '.join([m.get_full_name() for _, m in res])}")
            
        # 指定供应方
        provider = self.get_provider(provider_name)
        # 找不到供应方
        if not provider:
            provider = self.get_closest_provider(provider_name)
            msg = f"未找到供应方 {provider_name}"
            if provider:
                msg += f"，是否是 {provider.name}?"
            raise Exception(msg)
        provider.update_models()
        
        # 获取模型
        model: Optional[LlmModel] = None
        for m in provider.models:
            if m.name == model_name:
                model = m
                break
        # 找不到模型
        if not model:
            model = self.get_closest_model(model_name, provider)
            msg = f"未找到模型 {model_name}"
            if model:
                msg += f"，是否是 {model.get_full_name()}?"
            raise Exception(msg)

        return model
        

api_provider_mgr = ApiProviderManager([
    AiyydsApiProvider(),
    OpenrouterApiProvider(),
    SiliconflowApiProvider(),
    GoogleApiProvider(),
    NewApiApiProvider(),
    NewApi2ApiProvider(),
    DeepseekApiProvider(),
    YunWuApiProvider(),
    LongcatApiProvider(),
])
