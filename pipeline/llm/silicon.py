"""
SiliconFlow 多 API Key 并发请求客户端
支持多个 prompts 使用多个 API keys 并发发送请求
"""
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import requests
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_fixed

from global_config import SILICONFLOW_MODEL, TIMEOUT, TEMPERATURE
from pipeline.llm.base import BaseLLMClient
import traceback
import time
 

class SiliconModel(BaseLLMClient):
    """
    SiliconFlow 多 API Key 并发客户端
    
    ═══════════════════════════════════════════════════════════
    本类实现了 2 个核心方法：
    ═══════════════════════════════════════════════════════════
    
    1. get_available_api_keys() - 获取可用的 SiliconFlow API keys
    2. _get_completion_with_key() - 使用指定 key 调用 SiliconFlow API
    
    ═══════════════════════════════════════════════════════════
    基类自动提供（无需实现）：
    ═══════════════════════════════════════════════════════════
    
    - get_completion() - 单次请求
    - get_completion_json() - 单次 JSON 请求
    - batch_get_completions() - 批量并行请求
    - batch_get_completions_json() - 批量 JSON 请求
    
    ═══════════════════════════════════════════════════════════
    使用示例：
    ═══════════════════════════════════════════════════════════
    
        # 自动获取可用的 API keys
        client = SiliconModel()
        
        # 单次请求
        response = client.get_completion('介绍机器学习')
        
        # 批量并行请求（自动使用多个 keys）
        prompts = ['问题1', '问题2', '问题3', ...]
        responses = client.batch_get_completions(prompts)
        
        # 指定模型
        client = SiliconModel(model='deepseek-ai/DeepSeek-V3')
    """
    
    def __init__(self):
        """
        初始化多 Key 客户端（自动获取可用的 API keys）
        
        Args:
            model: 模型名称（默认使用配置文件中的模型）
        """
        # 调用父类的 __init__（会自动调用 get_available_api_keys）
        super().__init__()
        
        # 设置 SiliconFlow 特有的属性
        self.model_name = 'qwen3-80b-instruct'
        self.endpoint = 'https://api.siliconflow.cn/v1/chat/completions'
        
        print(f"🔑 已加载 {len(self.api_keys)} 个 API keys")
        print(f"🤖 使用模型: {self.model_name}")
    
 
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
    def _get_completion_with_key(
        self, 
        prompt: str, 
        api_key: str, 
        temperature: float = TEMPERATURE
    ) -> str:
        """
        使用指定的 API key 获取单个 prompt 的响应（自动重试3次）
        
        Args:
            prompt: 输入提示
            api_key: API key
            temperature: 温度参数
            
        Returns:
            str: 响应内容
            
        Raises:
            Exception: 如果3次重试后仍然失败
        """
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        payload: Dict = {
            'model': self.model_name,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': temperature,
            'stream': False,
        }
        
        resp = requests.post(
            self.endpoint, 
            headers=headers, 
            json=payload, 
            timeout=TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        if resp.status_code != 200:
            raise ValueError(f'{resp.status_code} {resp.text}')
        
        
        # OpenAI 风格返回
        choices = data.get('choices') or []
        if not choices:
            content = json.dumps(data, ensure_ascii=False)
        else:
            message = choices[0].get('message') or {}
            content = message.get('content', '')
        
        time.sleep(0.1)
        return content
    
    
    def is_available(self) -> bool:
        key = self.get_available_api_keys()[0]
        try:
            self._get_completion_with_key(prompt='你好', api_key=key)
        except:
            return False
    
    @classmethod
    def get_available_api_keys(cls) -> List[str]:
        """
        获取所有可用的 SiliconFlow API keys（并发检查）
        
        Returns:
            List[str]: 可用的 API key 列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        ALL_API_KEYS = [
            'sk-lnthpodxfopgszzvigunjavkzsozafcfnlorknkstrnxfpht', #淘宝的，余额只有15，先不用
        #    'sk-pbqzwhfskmkbeplvdvajknebvowgbgshtpmmchfqkjpmqenh', #淘宝的
        #    'sk-tslruvlmldahdffvszhsityglgxiqkacywhxdlayeuweyemr', #别人的
        #    'sk-udszotvrqjgscjpikxgddplbyxatpllkatnkzmsokbrxffkj', #我自己的
        #    'sk-pzldensjcmksymvbynbnvagpmgzmpvincsrxulkgquuuyoea', #补的
        #   'sk-cqwqkyssjumedtwnroiftfxxrcdprobagupblkjgkqbvladc', #补的
            'sk-rfoibqztsjjxrnufvqqwklbkbrzjmnezpyypozznkqpfpnya', #small
        ]
        
        print(f"   并发检查 {len(ALL_API_KEYS)} 个 API keys...")
        
        # 使用 ThreadPoolExecutor 并发检查
        usable_keys = []
        with ThreadPoolExecutor(max_workers=len(ALL_API_KEYS)) as executor:
            # 直接 submit get_silicon_user_info
            future_to_key = {executor.submit(get_silicon_user_info, key): key for key in ALL_API_KEYS}
            
            # 收集结果
            for future in as_completed(future_to_key):
                api_key = future_to_key[future]
                key_suffix = api_key[-8:] if len(api_key) > 8 else api_key
                
                try:
                    balance = future.result()
                    print(f'   ✅ ...{key_suffix} 余额: {balance:.2f}')
                    if balance > 0:
                        usable_keys.append(api_key)
                except Exception as e:
                    print(f'   ❌ ...{key_suffix} 检查失败: {str(e)[:50]}')
        
        return usable_keys

def get_silicon_user_info(api_key: str) -> Dict:
    headers = {
        'Authorization': f'Bearer {api_key}',
    }
    resp = requests.get(f'https://api.siliconflow.cn/v1/user/info', headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    
    data = resp.json()['data']
    balance = float(data.get('balance'))
    return balance
    
    # 以下方法由基类自动提供：
    # - get_completion: 使用第一个 API key
    # - get_completion_json: 自动解析 JSON
    # - batch_get_completions: 自动并行处理（ThreadPoolExecutor）
    # - batch_get_completions_json: 批量 JSON 解析


if __name__ == '__main__':
    # 测试示例
    print("=" * 60)
    print("SiliconFlow 多 Key 并发客户端测试")
    print("=" * 60)
    
    # 创建客户端（自动获取可用 keys）
    client = SiliconModel()
    

    import time
    

    start_time = time.time()
    # 测试多个 prompts
    test_prompts = [
        "用一句话介绍什么是机器学习",
        "用一句话介绍什么是深度学习",
        "用一句话介绍什么是神经网络",
        "用一句话介绍什么是自然语言处理",
        "用一句话介绍什么是计算机视觉",
        "用一句话介绍什么是强化学习",
    ]
    
    results = client.batch_get_completions(test_prompts, temperature=0.1, max_workers=2)
    
    # 打印结果
    for i, (prompt, response) in enumerate(zip(test_prompts, results), 1):
        print(f"\n{i}. Prompt: {prompt}")
        print(f"   Response: {response[:100]}...")  # 只显示前100个字符

    end_time = time.time()
    print(f"\n⏱️  Time taken: {end_time - start_time:.2f} seconds, max_workers=2")
    print('=' * 60)