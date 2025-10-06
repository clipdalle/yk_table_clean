# 语音厅考勤表 - 字段清洗

使用 Gemini API 清洗混乱的"主持"和"排麦人员"字段

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置
编辑 `CONFIG.py` 文件：
```python
# API Key
GOOGLE_API_KEY = "your-gemini-api-key"
GLM4_API_KEY = "your-glm4-api-key"

# 模型选择（'gemini' 或 'glm4'）
CURRENT_MODEL = 'glm4'  # 当前使用的模型

# LLM 参数
TEMPERATURE = 0.3   # 温度参数 (0.0-1.0)，越低越确定
BATCH_SIZE = 10     # 每批处理的行数
```

### 3. 运行（推荐）
```bash
python main.py
```
交互式菜单，选择操作

或者直接运行：
```bash
# 测试
python pipeline/test_clean_sample.py

# 批量处理
python pipeline/clean_pipeline.py
```

### 4. 查看结果
会生成两个文件：
- `output/主持打卡9.30_cleaned.xlsx` - 纯文本版
- `output/主持打卡9.30_cleaned_with_formula.xlsx` - 带公式版（推荐）

## 项目结构

```
smart_table/
├── main.py                       # 主入口（交互式菜单）
├── pipeline/                     # 代码文件夹
│   ├── llm_client.py            # 统一 LLM 调用接口
│   ├── gemini_bot.py            # Gemini API 客户端
│   ├── glm4_bot.py              # GLM-4 API 客户端
│   ├── prompts.py               # Prompt 模板
│   ├── clean_pipeline.py        # 批量清洗主流程
│   └── test_clean_sample.py     # 测试脚本
├── data/                         # 数据文件夹
├── output/                       # 输出文件夹
└── CONFIG.py                     # 配置文件
```

## 预期效果

- AI 准确率：85-95%
- 人工校验后：100%
- 处理时间：约 1 分钟（200条，每批10行，共20次API调用）
- 成本：几乎免费

## 新增列说明

清洗后会在原表右侧新增 7 列：

**主持相关（1列）：**
1. `主持人员列表_AI解析` - 清洗后的主持人员（竖线分隔）

**排麦相关（4列）：**
2. `排麦人员列表_AI解析` - 清洗后的排麦人员（竖线分隔）
3. `排麦出席人数_AI解析` - 实际解析到的人员数量
4. `排麦缺席人数_AI解析` - 从文本中读取的缺席数（如"缺2"）
5. `排麦置信度_AI解析` - 置信度（high/medium/low）

**错误信息（2列，在最右侧）：**
6. `主持错误_AI解析` - 主持字段解析错误
7. `排麦错误_AI解析` - 排麦字段解析错误

**注意：** 人员列表使用竖线 `|` 分隔，如：`王摆摆|九酱|珊珊`

**带公式版本额外功能：**
- Sheet2: 人员主持排麦统计（自动统计每个人的主持次数和排麦次数）