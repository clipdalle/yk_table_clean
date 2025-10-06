"""
Gemini API 调用（随机 Key 版本）
每次调用 get_completion 都随机选择一个 GOOGLE API KEY
依赖：pip install google-generativeai
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional
from google import genai

try:
    from CONFIG import get_google_api_key
except ImportError:
    # 如果无法导入CONFIG，使用环境变量
    def get_google_api_key():
        return os.getenv('GOOGLE_API_KEY', '')


class GeminiRandomClient:
    """每次调用随机选择 API Key 的 Gemini 客户端"""

    def __init__(self, model: str = 'gemini-2.5-flash'):
        self.model = model

    def get_completion(self, prompt: str, temperature: float = 0.7, api_key: Optional[str] = None) -> str:
        """
        使用随机（或指定）API Key 调用 Gemini 接口

        Args:
            prompt: 提示词
            temperature: 温度参数 (0-1)
            api_key: 可选，若传入则使用该 key，否则随机选择

        Returns:
            模型返回文本或错误信息
        """
        try:
            key = api_key or get_google_api_key()
            os.environ['GOOGLE_API_KEY'] = key
            client = genai.Client(api_key=key)

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={'temperature': temperature}
            )
            return response.text
        except Exception as e:
            return f"Error calling Gemini API: {e}"


# -------- 测试示例 --------
if __name__ == '__main__':
    bot = GeminiRandomClient()
    demo = "用一句话介绍什么是机器学习"
    print("--- 使用随机 Key 调用 Gemini ---")
    print(bot.get_completion(demo))


