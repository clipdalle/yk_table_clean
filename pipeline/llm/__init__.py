"""
LLM 模块
包含各种 LLM 客户端的实现
"""

from .base import BaseLLMClient
from .llm_client import get_completion, _get_client
from .gemini_bot import GeminiClient
from .gemini_random_bot import GeminiRandomClient
from .glm4_bot import GLM4Client
from .silicon import SiliconModel

__all__ = [
    # 基类
    'BaseLLMClient',
    # 具体实现
    'get_completion',
    '_get_client', 
    'GeminiClient',
    'GeminiRandomClient',
    'GLM4Client',
    'SiliconModel',
]
