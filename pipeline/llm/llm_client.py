"""
统一的 LLM 调用接口
根据 CONFIG.CURRENT_MODEL 自动选择使用 Gemini 或 GLM-4
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from CONFIG import CURRENT_MODEL


# 全局客户端实例（懒加载）



def _get_client(model_name: str = ''):
    """获取 LLM 客户端实例（单例模式）"""

    if model_name == 'gemini':
        from .gemini_bot import GeminiClient
        _client = GeminiClient()
    elif model_name == 'random_gemini':
        from .gemini_random_bot import GeminiRandomClient
        _client = GeminiRandomClient()
    elif model_name == 'glm4':
        from .glm4_bot import GLM4Client
        _client = GLM4Client()
    elif model_name == 'silicon_ds_v3':
        from .silicon import SiliconModel
        _client = SiliconModel(model='deepseek-ai/DeepSeek-V3.1-Terminus')
    elif model_name == 'silicon_ds_r1':
        from .silicon import SiliconModel
        _client = SiliconModel(model='deepseek-ai/DeepSeek-R1')
    #Qwen/Qwen3-Omni-30B-A3B-Instruct
    elif model_name == 'silicon_ds_omni':
        from .silicon import SiliconModel
        _client = SiliconModel(model='Qwen/Qwen3-Omni-30B-A3B-Instruct')
    elif model_name == 'silicon_qwen3_80B':
        from .silicon import SiliconModel
        _client = SiliconModel(model='Qwen/Qwen3-Next-80B-A3B-Instruct')
    else:
        raise ValueError(f"未知的模型类型: {model_name}，请在 CONFIG.py 中设置为 'gemini'、'random_gemini'、'glm4' 或 'silicon_ds_v3'")

    return _client


def get_completion(prompt: str, temperature: float = 0.7, model_name: str = '') -> str:
    """
    统一的 LLM 调用接口
    根据 CONFIG.CURRENT_MODEL 自动选择模型
    
    Args:
        prompt: 输入的提示词
        temperature: 温度参数，0-1之间
    
    Returns:
        模型返回的文本内容
    """
    client = _get_client(model_name)
    return client.get_completion(prompt, temperature=temperature, timeout=TIMEOUT)


if __name__ == '__main__':
    print(f"当前使用模型: {CURRENT_MODEL}")
    
    test_prompt = "用一句话介绍什么是机器学习"
    result = get_completion(test_prompt)
    print(f'\n--- {CURRENT_MODEL.upper()} 回复 ---')
    print(result)
