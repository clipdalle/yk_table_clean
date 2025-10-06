"""
GLM-4 API 调用（类封装）
安装依赖：pip install requests
"""

import requests
import json
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from CONFIG import GLM4_API_KEY
except ImportError:
    # 如果无法导入CONFIG，使用环境变量
    GLM4_API_KEY = os.getenv('GLM4_API_KEY', '')


class GLM4Client:
    """GLM-4 API 客户端"""
    
    def __init__(
        self, 
        api_key: str = GLM4_API_KEY, 
        model: str = 'glm-4-plus',
        api_url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    ):
        """
        初始化 GLM-4 客户端
        
        Args:
            api_key: API Key
            model: 模型名称（glm-4-flash, glm-4-plus, glm-4-air 等）
            api_url: API 地址
        """
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
    
    def get_completion(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        调用 GLM-4 API
        
        Args:
            prompt: 输入的提示词
            temperature: 温度参数，0-1之间，越高越随机
            max_tokens: 最大生成 token 数
        
        Returns:
            模型返回的文本内容
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
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
                return f"Error: 未找到返回内容 - {result}"
        
        except requests.exceptions.RequestException as e:
            return f"Error calling GLM-4 API: {e}"
        
        except Exception as e:
            return f"Error: {e}"


# -------- 测试示例 --------
if __name__ == '__main__':
    client = GLM4Client()
    
    # 测试简单对话
    test_prompt = "用一句话介绍什么是机器学习"
    result = client.get_completion(test_prompt)
    print('--- GLM-4 回复 ---')
    print(result)
