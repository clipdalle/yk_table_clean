"""
SiliconFlow DeepSeek V3 客户端
参考 cURL:
curl --request POST \
  --url https://api.siliconflow.cn/v1/chat/completions \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '{
  "model": "Qwen/QwQ-32B",
  "messages": [
    {"role": "user", "content": "..."}
  ]
}'
"""
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import requests
from typing import List, Dict

from global_config import SILICONFLOW_API_KEY, SILICONFLOW_MODEL, TIMEOUT, TEMPERATURE

class SiliconModel:
    """SiliconFlow DeepSeek V3 API 客户端"""

    def __init__(self, api_key: str = SILICONFLOW_API_KEY, model: str = SILICONFLOW_MODEL):
        self.api_key = api_key
        self.model = model
        self.endpoint = 'https://api.siliconflow.cn/v1/chat/completions'

    def get_completion(self, prompt: str, temperature: float = TEMPERATURE) -> str:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        payload: Dict = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': temperature,
            'stream': False,
        }

        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            # OpenAI 风格返回
            choices = data.get('choices') or []
            if not choices:
                return json.dumps(data, ensure_ascii=False)
            message = choices[0].get('message') or {}
            return message.get('content', '')
        except requests.HTTPError as http_err:
            # 输出更详细的错误信息，便于定位 403 等权限/模型问题
            detail = ''
            try:
                detail = f" | body={resp.text}"
            except Exception:
                pass
            return f"Error calling SiliconFlow API: {http_err}{detail}"
        except Exception as e:
            return f"Error calling SiliconFlow API: {e}"


if __name__ == '__main__':
    client = SiliconModel()
    print(client.get_completion('用一句话介绍什么是机器学习', temperature=0.1))


