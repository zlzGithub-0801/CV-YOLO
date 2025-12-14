# 使用 API 调用的快速安装指南

如果你想使用 API 调用而不是本地模型，只需要安装以下依赖：

## 必需依赖

```bash
# YOLO
pip install ultralytics opencv-python

# CLIP (中文)
pip install cn-clip pillow numpy

# API 调用库（二选一）
pip install dashscope  # 阿里云通义千问
# 或
pip install openai     # OpenAI 兼容接口

# 工具库
pip install matplotlib tqdm
```

## 完整安装（包含本地模型支持）

```bash
pip install -r requirements.txt
```

## 配置 API Key

编辑 `config.py` 文件：

### 使用阿里云通义千问

```python
LLM_USE_API = True
LLM_API_TYPE = "dashscope"
DASHSCOPE_API_KEY = "sk-xxxxx"  # 你的 API Key
DASHSCOPE_MODEL = "qwen-turbo"  # 或 qwen-plus, qwen-max
```

获取 API Key：
1. 访问 https://dashscope.aliyun.com/
2. 注册/登录阿里云账号
3. 开通 DashScope 服务
4. 获取 API Key

### 使用 OpenAI 兼容接口

```python
LLM_USE_API = True
LLM_API_TYPE = "openai"
OPENAI_API_KEY = "sk-xxxxx"
OPENAI_API_BASE = "https://api.openai.com/v1"  # 或其他兼容服务
OPENAI_MODEL = "gpt-3.5-turbo"
```

## 运行测试

```bash
python demo_simple.py test.jpg
```
