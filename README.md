# 麦序数据清洗 Flask 应用

一个基于 Flask 的麦序数据清洗 Web 应用，支持上传全量名单和麦序Excel文件，在线处理并返回结果。

## 功能特点

- 🎤 **智能解析**：使用 LLM 自动解析麦序数据
- 📋 **名单匹配**：基于全量名单进行人名标准化和匹配
- 🎨 **美观界面**：现代化的 Web 界面设计
- ☁️ **云端部署**：支持 Vercel 部署
- 📊 **实时处理**：在线处理，无需本地安装

## 技术栈

- **后端**：Flask + Python
- **前端**：HTML5 + CSS3 + JavaScript
- **数据处理**：pandas + openpyxl
- **AI 处理**：Google Gemini / GLM-4 / DeepSeek V3
- **部署**：Vercel

## 项目结构

```
├── app.py                 # Flask 主应用
├── templates/
│   └── index.html        # 前端页面
├── pipeline/              # 核心业务逻辑
│   ├── clean_pipeline_v3.py
│   ├── llm_client.py
│   ├── prompts_v3.py
│   └── ...
├── CONFIG.py             # 配置文件
├── requirements.txt      # Python 依赖
├── vercel.json          # Vercel 配置
└── README.md
```

## 使用方法

### 1. 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
python app.py
```

访问 `http://localhost:5000` 即可使用。

### 2. Vercel 部署

1. 将代码推送到 GitHub
2. 在 Vercel 中导入项目
3. 配置环境变量（API Keys）
4. 部署完成

## 使用流程

1. **上传全量名单**：上传包含所有可能人名的 `.txt` 文件
2. **上传麦序Excel**：上传需要处理的麦序数据 `.xlsx` 文件
3. **开始处理**：点击处理按钮，系统自动解析数据
4. **下载结果**：处理完成后下载清洗后的 Excel 文件

## 核心功能

- **人名标准化**：自动处理大小写、装饰符号
- **智能解析**：使用 LLM 解析混乱的麦序文本
- **数据清洗**：提取主持人员、排麦人员、缺席人数等
- **错误标记**：标记未知人名和低置信度数据
- **Excel 输出**：生成带公式和着色的 Excel 文件

## 环境变量

在 Vercel 中配置以下环境变量：

```
GOOGLE_API_KEY=your_gemini_api_key
GLM4_API_KEY=your_glm4_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
```

## 注意事项

- 文件大小限制：建议单个文件不超过 10MB
- 处理时间：根据数据量可能需要 1-5 分钟
- 数据安全：文件仅在处理期间临时存储，不会永久保存
- API 限制：注意各 LLM 服务的 API 调用限制

## 许可证

MIT License