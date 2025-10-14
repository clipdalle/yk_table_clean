"""
GLM-4 API 调用（类封装）
安装依赖：pip install requests
"""

import requests
import json
import sys
import os
from typing import List
from tenacity import retry, stop_after_attempt, wait_fixed

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)


from pipeline.llm.base import BaseLLMClient


class GLM4Client(BaseLLMClient):
    """
    GLM-4 API 客户端
    
    ═══════════════════════════════════════════════════════════
    本类实现了 2 个核心方法：
    ═══════════════════════════════════════════════════════════
    
    1. get_available_api_keys() - 返回配置的 GLM-4 API key
    2. _get_completion_with_key() - 使用指定 key 调用 GLM-4 API
    
    ═══════════════════════════════════════════════════════════
    基类自动提供（无需实现）：
    ═══════════════════════════════════════════════════════════
    
    - get_completion() - 单次请求
    - get_completion_json() - 单次 JSON 请求
    - batch_get_completions() - 批量并行请求
    - batch_get_completions_json() - 批量 JSON 请求
    """
    
    def __init__(
        self, 
        max_tokens: int = 2000
    ):
        """
        初始化 GLM-4 客户端（自动获取可用的 API keys）
        
        Args:
            model: 模型名称（glm-4-flash, glm-4-plus, glm-4-air 等）
            api_url: API 地址
            max_tokens: 最大生成 token 数
        """
        super().__init__()  # 自动获取 API keys
        
        self.model_name = "glm-4-plus"
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.max_tokens = max_tokens
        
        print(f"🔑 已加载 {len(self.api_keys)} 个 API keys")
        print(f"🤖 使用模型: {self.model_name}")
    
    @classmethod
    def get_available_api_keys(cls) -> List[str]:
        """
        获取可用的 GLM-4 API keys
        
        Returns:
            List[str]: 可用的 API key 列表
        """
        # GLM-4 通常只有一个 key，从配置文件读取
        GLM4_API_KEY_LIST = [
            '4e51ecf7fcae4bed90edf3b9d4a026a7.XdH96shx1QedVUqb',  # 智谱 GLM-4 API Key
            'f861430f05c0415a915d2c9d7b1e06e5.CkSoQmKgQySRlPDc'  # 智谱 GLM-4 API Key
        ]
        return GLM4_API_KEY_LIST
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
    def _get_completion_with_key(
        self, 
        prompt: str, 
        api_key: str,
        temperature: float = 0.7
    ) -> str:
        """
        使用指定的 API key 调用 GLM-4 API（自动重试3次）
        
        Args:
            prompt: 输入的提示词
            api_key: 要使用的 API key
            temperature: 温度参数，0-1之间，越高越随机
        
        Returns:
            str: 模型返回的文本内容
            
        Raises:
            Exception: 如果3次重试后仍然失败
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "do_sample": True
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        
        result = response.json()
        
        # 提取返回的文本
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise ValueError(f"未找到返回内容 - {result}")
    
    # 以下方法由基类自动提供：
    # - get_completion() - 单次请求
    # - get_completion_json() - 单次 JSON 请求
    # - batch_get_completions() - 批量并行请求
    # - batch_get_completions_json() - 批量 JSON 请求


# -------- 测试示例 --------
if __name__ == '__main__':
    # 自动获取可用的 API keys
    client = GLM4Client()
    
    # # 测试单次请求
    # test_prompt = "用一句话介绍什么是机器学习"
    # result = client.get_completion(test_prompt)
    # print('\n--- GLM-4 单次请求 ---')
    # print(result)
    
    # 测试批量请求
    test_prompts = [
        "用一句话介绍什么是深度学习",
        "用一句话介绍什么是神经网络",
        "用一句话介绍什么是自然语言处理"
    ]
    results = client.batch_get_completions(test_prompts)
    print('\n--- GLM-4 批量请求 ---')
    for i, (prompt, response) in enumerate(zip(test_prompts, results), 1):
        print(f"{i}. {prompt}")
        print(f"   {response}\n")
