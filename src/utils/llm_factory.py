import json
import logging
import os
from typing import Dict, List, Tuple, Callable, Any, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError, SecretStr

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config.llm_config import LLMConfig
from src.config.loader import Config

TModel = TypeVar("TModel", bound=BaseModel)

# Mock ArgOptionsGeneric for type hinting if needed, or just use Union
ArgOptionsGeneric = Any 

def _normalize_settings_kwargs(setting: Any) -> Dict[str, Any]:
    """
    Normalize LLM settings to kwargs dict.
    - Supports Pydantic v2 models (model_dump), Pydantic v1 models (.dict()), and plain dicts.
    - Filters out None values.
    """
    if setting is None:
        return {}

    try:
        if isinstance(setting, BaseModel):
            # Prefer v2 model_dump if available
            try:
                return setting.model_dump(exclude_none=True)
            except Exception:
                # Fallback for pydantic v1
                return setting.model_dump(exclude_none=True)
    except Exception:
        # If BaseModel import/type check fails for any reason, continue
        pass

    if isinstance(setting, dict):
        return {k: v for k, v in setting.items() if v is not None}

    # Unknown type, ignore
    return {}


def extract_json_from_text(text: str) -> str:
    """
    从 LLM 的自然语言回复中提取 JSON 字符串（通用工具）。

    优先策略：
    - 优先从最后一个 ```json ... ``` 代码块中提取
    - 若没有代码块，则从整段文本中搜索最后一对花括号 { ... }
    """
    if not text:
        raise ValueError("Empty text, cannot extract JSON.")

    # 1. 优先找 ```json 代码块（从后往前找，取最后一个）
    fence = "```"
    json_fence = "```json"
    last_json_fence = text.rfind(json_fence)
    if last_json_fence != -1:
        # 从 ```json 开始，往后找下一个 ```
        start = last_json_fence + len(json_fence)
        end = text.find(fence, start)
        if end != -1:
            json_block = text[start:end].strip()
            if json_block:
                return json_block

    # 2. 退化策略：直接在全文里找最后一对花括号
    left = text.rfind("{")
    right = text.rfind("}")
    if left == -1 or right == -1 or right <= left:
        raise ValueError("No JSON object found in text.")
    return text[left : right + 1]


def parse_json_text_to_model(text: str, model_cls: Type[TModel]) -> TModel:
    """
    通用工具：从 LLM 文本回复中提取 JSON，并解析为指定的 Pydantic BaseModel（含 RootModel）。

    - `text`: LLM 返回的整段文本（通常来自 message.content）
    - `model_cls`: 目标 Pydantic 模型类（包括 RootModel 子类）

    用法示例：

        mapping = parse_json_text_to_model(llm_text, AffiliationMap)
    """
    try:
        json_str = extract_json_from_text(text)
    except Exception as e:
        logging.error(f"Failed to extract JSON from LLM text: {type(e).__name__}: {e}")
        raise

    try:
        data = json.loads(json_str)
    except Exception as e:
        logging.error(f"Failed to parse JSON from LLM text: {type(e).__name__}: {e}")
        raise

    try:
        # 兼容 Pydantic v2 的 model_validate / v1 的 parse_obj
        if hasattr(model_cls, "model_validate"):
            return model_cls.model_validate(data)  # type: ignore[return-value]
        return model_cls.parse_obj(data)  # type: ignore[return-value,attr-defined]
    except Exception as e:
        logging.error(
            f"Failed to convert JSON to model {getattr(model_cls, '__name__', str(model_cls))}: "
            f"{type(e).__name__}: {e}"
        )
        raise


def init_langchain_models_from_llm_config(
    llm_config: List[Union[LLMConfig, Any]],
) -> Tuple[Dict[str, BaseChatModel], Optional[BaseChatModel]]:
    """
    初始化 LangChain 所需的聊天模型集合，并返回默认模型。
    - 输入为 LLMConfig 列表（来自 config.yaml）
    - 优先使用 LangChain 的 init_chat_model，根据供应商自动选择后端
    - 失败时回退到 ChatOpenAI（支持 OpenAI 兼容接口）
    - 返回：{"type:model": BaseChatModel}, default_model
    """
    models: Dict[str, BaseChatModel] = {}
    default_model: Optional[BaseChatModel] = None

    def _extract_fields(item: Any) -> tuple[str, str, Optional[str], Optional[str], Any]:
        """
        Extract (provider, model, base_url, api_key, setting) from either LLMConfig
        or ArgOptionsGeneric[LLMConfig]-like objects.
        """
        # ArgOptionsGeneric
        if hasattr(item, "params") and hasattr(item, "type"):
            provider = getattr(item, "type")
            params = getattr(item, "params")
            model = getattr(params, "model", None)
            base_url = getattr(params, "base_url", None)
            api_key = getattr(params, "api_key", None)
            setting = getattr(params, "setting", None)
            return provider, model, base_url, api_key, setting
        # Plain LLMConfig
        provider = getattr(item, "type", None)
        model = getattr(item, "model", None)
        base_url = getattr(item, "base_url", None)
        api_key = getattr(item, "api_key", None)
        setting = getattr(item, "setting", None)
        return provider, model, base_url, api_key, setting

    for each in llm_config:
        provider, model_name, base_url, api_key, setting = _extract_fields(each)
        if not provider or not model_name:
            logging.error(f"Invalid LLM item, missing provider/model: {each}")
            continue
        key = f"{provider}:{model_name}"
        settings_kwargs = _normalize_settings_kwargs(setting)
        settings_kwargs.setdefault("timeout", 300)
        init_kwargs = dict(
            model_provider=provider,
            model=model_name,
            **settings_kwargs,
        )
        if api_key:
            init_kwargs["api_key"] = api_key
        if base_url:
            init_kwargs["base_url"] = base_url
        try:
            model = init_chat_model(**init_kwargs)
            models[key] = model
            if default_model is None:
                default_model = model
        except Exception as e:
            logging.warning(f"Failed to init model {key} with init_chat_model: {e}. Trying fallback to ChatOpenAI.")
            try:
                fallback_kwargs = dict(
                    model=model_name,
                    **settings_kwargs,
                )
                if api_key:
                    fallback_kwargs["api_key"] = api_key
                if base_url:
                    fallback_kwargs["base_url"] = base_url
                model = ChatOpenAI(**fallback_kwargs)
                models[key] = model
                if default_model is None:
                    default_model = model
            except Exception as e2:
                logging.error(f"Failed to init fallback model {key}: {e2}")

    return models, default_model
