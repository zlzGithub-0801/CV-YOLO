# YOLO + LLM 图像描述生成系统

基于 **Socratic Models** (arXiv 2204.00598) 论文思想的图像描述生成系统。

## 📋 项目简介

本项目整合了三个强大的AI模型：
- **YOLO (YOLOv8)**: 物体检测，提取场景信息
- **Qwen-LLM**: 大语言模型，生成候选描述
- **CLIP**: 图像-文本匹配，选择最佳描述

### 工作流程

```
输入图像
   ↓
┌─────────────────┐
│  YOLO 检测器    │ → 物体、位置、场景
└─────────────────┘
   ↓
┌─────────────────┐
│  LLM 生成器     │ → 20个候选描述
└─────────────────┘
   ↓
┌─────────────────┐
│  CLIP 排序器    │ → 计算相似度
└─────────────────┘
   ↓
输出最佳描述
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- CUDA (可选，推荐用于加速)

### 2. 安装依赖

**选项 A: 使用 API 调用（推荐，无需下载大模型）**

```bash
# 基础依赖
pip install ultralytics opencv-python cn-clip pillow numpy matplotlib tqdm

# 安装 API 库（二选一）
pip install dashscope  # 阿里云通义千问
# 或
pip install openai     # OpenAI 兼容接口
```

然后配置 API Key（见下文 "API 配置" 部分）

**选项 B: 使用本地模型（需要下载 ~4GB 模型）**

```bash
pip install -r requirements.txt
```

### 3. 配置 API（如果选择 API 调用）

编辑 `config.py` 文件：

#### 使用阿里云通义千问（推荐）

```python
LLM_USE_API = True
LLM_API_TYPE = "dashscope"
DASHSCOPE_API_KEY = "sk-xxxxx"  # 填入你的 API Key
DASHSCOPE_MODEL = "qwen-turbo"  # 或 qwen-plus, qwen-max
```

**获取 API Key**: 访问 https://dashscope.aliyun.com/

#### 使用 OpenAI 兼容接口

```python
LLM_USE_API = True
LLM_API_TYPE = "openai"
OPENAI_API_KEY = "sk-xxxxx"
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-3.5-turbo"
```

### 4. 下载模型（如果使用本地模型）

YOLO 和 CLIP 模型会自动下载。

对于 Qwen-LLM，首次运行会自动从 HuggingFace 下载，或手动下载：
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat")
```

### 4. 运行示例

```bash
# 基本用法
python 11.py test.jpg

# 生成10个候选并可视化
python 11.py test.jpg --num_candidates 10 --visualize

# 保存结果
python 11.py test.jpg --save_result --visualize
```

## 📁 项目结构

```
yolo_LLM/
├── 11.py                 # 主程序
├── config.py             # 配置文件
├── yolo_detector.py      # YOLO检测模块
├── llm_generator.py      # LLM生成模块
├── clip_ranker.py        # CLIP排序模块
├── utils.py              # 工具函数
├── requirements.txt      # 依赖清单
├── README.md             # 说明文档
└── outputs/              # 输出目录
```

## ⚙️ 配置说明

编辑 `config.py` 可以调整：

- **YOLO 模型**: `YOLO_MODEL = "yolov8n.pt"` (可选: yolov8s, yolov8m)
- **LLM 模型**: `LLM_MODEL_NAME = "Qwen/Qwen-7B-Chat"`
- **CLIP 模型**: `CLIP_MODEL_TYPE = "chinese-clip"`
- **候选数量**: `NUM_CANDIDATES = 20`
- **描述长度**: `MIN_CAPTION_LENGTH = 15`, `MAX_CAPTION_LENGTH = 30`

## 📊 使用示例

### 单独测试各模块

```bash
# 测试 YOLO
python yolo_detector.py test.jpg

# 测试 LLM
python llm_generator.py

# 测试 CLIP
python clip_ranker.py test.jpg
```

### 完整流程

```python
from yolo_detector import YOLODetector
from llm_generator import LLMGenerator
from clip_ranker import CLIPRanker

# 初始化
yolo = YOLODetector()
llm = LLMGenerator()
clip = CLIPRanker()

# 生成描述
yolo_result = yolo.detect("test.jpg")
candidates = llm.generate_candidates(yolo_result)
ranked = clip.rank_captions("test.jpg", candidates)

print(f"最佳描述: {ranked[0][0]}")
```

## 💡 性能优化建议

### 如果没有GPU
1. 使用较小的 LLM 模型: `Qwen-1.8B-Chat`
2. 减少候选数量: `--num_candidates 5`
3. 使用量化模型

### 加速推理
1. 使用 GPU: 自动检测并使用 CUDA
2. 批量处理: 一次处理多张图像

## 🔧 故障排除

### 常见问题

**Q: 提示找不到模块？**
```bash
pip install -r requirements.txt
```

**Q: YOLO下载太慢？**
手动下载 yolov8n.pt 放到项目目录

**Q: LLM内存不足？**
使用更小的模型或量化版本

**Q: CLIP相似度都很低？**
检查是否使用了正确的中文/英文CLIP模型

## 📝 论文参考

本项目基于以下论文的思想：

**Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language**
- arXiv: 2204.00598
- 核心思想: 通过语言作为"桥梁"，组合多个预训练模型完成复杂的多模态任务

## 🎯 后续改进方向

- [ ] 支持批量处理
- [ ] 添加 Web UI (Gradio)
- [ ] 实现评估指标 (BLEU, CIDEr)
- [ ] 支持更多 CLIP 模型
- [ ] 添加模型缓存机制
- [ ] 支持视频描述生成

## 👥 作者

计算机视觉课程大作业项目

## 📄 许可

MIT License
