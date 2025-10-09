"""
配置文件
"""
import random 

# ========== API 配置 ==========
GOOGLE_API_KEY = 'AIzaSyDftWe8NeME53dZTzONSGW-gkSUVyY6_50'
TAOBAO_KEYS = [
#    'AIzaSyAqgs_X3PPWYDD7cQDNafkX5hKpqSt2e20',
#    'AIzaSyDZhidu9YoPXPXbV1giKmJk10bn7S2rPLw',
#    'AIzaSyCdZu2JXVjMZmOkOT4KG4bRZRtBZoLsZFA',
    'AIzaSyADNoFl4SWf07wKP8xC5O5f_XtsN4Tyf-8',
    'AIzaSyAd6s-r4o9oW4QhtKwx74YOd6os26x5x6I',
    'AIzaSyC9dYqUMfHky_if3o8r-_9hFgvtH5NAx4g',
#    'AIzaSyB18fGPzRTaT04PEjBj4O_OIeuNWYDgWXM',
#    'AIzaSyDeZXdMYvNdQyj8E36IxfYgxl4LnkvKHYU',
    'AIzaSyDucQBJYYDLc_1b3hz9J09PTO_5waBb2RA',
    'AIzaSyAKjhdh-voSll0YoPkC-uhKlYiJ_M3gR9M'
]

'''
Error calling Gemini API: 403 PERMISSION_DENIED. {'error': {'code': 403, 'message': "Permission denied: Consumer 'api_key:AIzaSyB18fGPzRTaT04PEjBj4O_OIeuNWYDgWXM' has been suspended.", 'status': 'PERMISSION_DENIED', 'details': [{'@type': 'type.googleapis.com/google.rpc.ErrorInfo', 'reason': 'CONSUMER_SUSPENDED', 'domain': 'googleapis.com', 'metadata': {'containerInfo': 'api_key:AIzaSyB18fGPzRTaT04PEjBj4O_OIeuNWYDgWXM', 'consumer': 'projects/1063164255', 'service': 'generativelanguage.googleapis.
Traceback (most recent call last):
'''
def get_google_api_key():
    return GOOGLE_API_KEY
    return random.choice(TAOBAO_KEYS)

GLM4_API_KEY = '4e51ecf7fcae4bed90edf3b9d4a026a7.XdH96shx1QedVUqb'  # 智谱 GLM-4 API Key

# SiliconFlow DeepSeek V3
SILICONFLOW_API_KEY = 'sk-udszotvrqjgscjpikxgddplbyxatpllkatnkzmsokbrxffkj'
SILICONFLOW_MODEL = 'deepseek-ai/DeepSeek-V3.1-Terminus'#'deepseek-ai/DeepSeek-V3'

# ========== 模型选择 ==========
# 可选值：'gemini' 或 'glm4' 或者 'random_gemini' 或者 "silicon_ds_v3"
CURRENT_MODEL = 'glm4' #'silicon_ds_v3'  # 当前使用的模型
#CURRENT_MODEL = 'silicon_ds_omni' #deepseek-ai/DeepSeek-R1
#CURRENT_MODEL = 'silicon_qwen3_80B'
STRICT_DATE_FILTER = True
# ========== LLM 调用参数 ==========

# 温度参数 (0.0 - 1.0)
# - 0.0-0.3: 更确定、一致的输出，适合数据提取任务
# - 0.4-0.7: 平衡创造性和一致性
# - 0.8-1.0: 更有创意、随机的输出
TEMPERATURE = 0.05

# 批次大小
# - 每批处理的行数，一次 API 调用处理多行数据
# - 建议值: 5-20 行
# - 太小: API 调用次数多，速度慢
# - 太大: 单次处理时间长，容易超时或出错
BATCH_SIZE = 10

# ========== 数据筛选 ==========

# 厅号筛选已移至前端页面选择，不再使用全局配置

TIMEOUT = 120
 

COLS_CONFIG = [
    {'col_name':'提交者（自动）', 'col_color':'default'},
    {'col_name':'提交时间（自动）', 'col_color':'default'},
    {'col_name':'日期（必填）', 'col_color':'default'},
    {'col_name':'厅号（必填）', 'col_color':'default'},
    {'col_name':'主持（必填）', 'col_color':'F5DEB3'},  
    {'col_name':'主持时间（必填）', 'col_color':'default'},  
    {'col_name':'主持人员列表_AI解析', 'col_color':'90EE90'},  # 浅绿色
    {'col_name':'排麦人员（必填）', 'col_color':'F5DEB3'},
    {'col_name':'排麦人员列表_AI解析', 'col_color':'90EE90'},  # 浅绿色
    {'col_name':'排麦出席人数_AI解析', 'col_color':'default'},
    {'col_name':'排麦缺席人数_AI解析', 'col_color':'default'},
    {'col_name':'排麦置信度_AI解析', 'col_color':'default'},
    {'col_name':'填坑（必填）', 'col_color':'default'},
    {'col_name':'黑麦 公屏不互动', 'col_color':'default'},
    {'col_name':'主持错误_AI解析', 'col_color':'default'},
    {'col_name':'排麦错误_AI解析', 'col_color':'default'}
]