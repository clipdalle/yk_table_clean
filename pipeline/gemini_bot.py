"""
Gemini API 调用（类封装）
安装依赖：pip install google-generativeai
"""

import os
import sys
sys.path.append('.')

from CONFIG import GOOGLE_API_KEY, get_google_api_key
from google import genai


class GeminiClient:
    """Gemini API 客户端"""
    
    def __init__(self, api_key: str = get_google_api_key(), model: str = 'gemini-2.5-flash'):
        """
        初始化 Gemini 客户端
        
        Args:
            api_key: API Key
            model: 模型名称
        """
        self.api_key = api_key
        self.model = model
        os.environ['GOOGLE_API_KEY'] = api_key
        self.client = genai.Client(api_key=api_key)
    
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        """
        调用 Gemini API
        
        Args:
            prompt: 输入的提示词
            temperature: 温度参数，0-1之间，越高越随机
        
        Returns:
            模型返回的文本内容
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={'temperature': temperature}
            )
            
            return response.text
        
        except Exception as e:
            return f"Error calling Gemini API: {e}"


# -------- 测试示例 --------
if __name__ == '__main__':
    client = GeminiClient()
    
    # 测试简单对话
    test_prompt = "用一句话介绍什么是机器学习"
    result = client.get_completion(test_prompt)
    print('--- Gemini 回复 ---')
    print(result)
 