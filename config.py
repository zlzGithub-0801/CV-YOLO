"""
配置文件：模型路径、参数设置
"""

import os

# ============ 模型配置 ============

# YOLO 配置
YOLO_MODEL = "yolov8n.pt"  # 可选: yolov8s.pt, yolov8m.pt
YOLO_CONF_THRESHOLD = 0.25  # 置信度阈值
YOLO_IOU_THRESHOLD = 0.45   # NMS IoU阈值

# LLM 配置 (Qwen)
LLM_USE_API = True  # 是否使用API调用（True）还是本地模型（False）

# 本地模型配置（当 LLM_USE_API=False 时使用）
# LLM_MODEL_NAME = "Qwen/Qwen-7B-Chat"  # 可选: Qwen-14B-Chat, Qwen-1.8B-Chat
# LLM_DEVICE = "cuda"  # 或 "cpu"

# API 配置（当 LLM_USE_API=True 时使用）
LLM_API_TYPE = "openai"  # 改为使用 OpenAI 兼容接口

# 阿里云通义千问 API 配置（暂时不用）
DASHSCOPE_API_KEY = "sk-7effebc913014a5eac8f2a0e760bd36b"
DASHSCOPE_MODEL = "qwen-max"

# OpenAI 兼容 API 配置 ⚠️ 请填入你的千问API信息
OPENAI_API_KEY = "sk-7effebc913014a5eac8f2a0e760bd36b"  # 你的API Key
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # ⚠️ 例如: https://your-api-url.com/v1
OPENAI_MODEL = "qwen-vl-max"  # 你的模型名称

# 通用生成参数
LLM_MAX_LENGTH = 2048
LLM_TEMPERATURE = 0.8  # 生成多样性
LLM_TOP_P = 0.9
LLM_TOP_K = 50

# CLIP 配置
CLIP_MODEL_TYPE = "chinese-clip"  # 修改为 openai-clip（因为 chinese-clip 在 Windows 编译失败）
CLIP_MODEL_NAME = "ViT-B-16"  # OpenAI CLIP 模型
CLIP_DOWNLOAD_ROOT = "models"  # clip 模型下载路径，如果没有会创造该路径

# ============ 生成配置 ============

# 候选描述数量
NUM_CANDIDATES = 3  # 生成的候选字幕数量
MAX_CAPTION_LENGTH = 100  # 字幕最大长度（字）
MIN_CAPTION_LENGTH = 50  # 字幕最小长度（字）

# ============ 位置映射 ============

# 将边界框坐标映射为位置描述
def get_position_description(x_center, y_center):
    """
    根据物体中心坐标返回位置描述
    x_center, y_center: 归一化坐标 (0-1)
    """
    # 水平位置
    if x_center < 0.33:
        h_pos = "画面左侧"
    elif x_center < 0.67:
        h_pos = "画面中央"
    else:
        h_pos = "画面右侧"
    
    # 垂直位置
    if y_center < 0.33:
        v_pos = "上方"
    elif y_center < 0.67:
        v_pos = "中部"
    else:
        v_pos = "下方"
    
    return f"{h_pos}{v_pos}"

# ============ YOLO 类别中文映射 ============

# COCO 80类的中文翻译
YOLO_CLASS_NAMES_ZH = {
    'person': '人', 'bicycle': '自行车', 'car': '汽车', 'motorcycle': '摩托车',
    'airplane': '飞机', 'bus': '公交车', 'train': '火车', 'truck': '卡车',
    'boat': '船', 'traffic light': '红绿灯', 'fire hydrant': '消防栓',
    'stop sign': '停车标志', 'parking meter': '停车计时器', 'bench': '长椅',
    'bird': '鸟', 'cat': '猫', 'dog': '狗', 'horse': '马', 'sheep': '羊',
    'cow': '牛', 'elephant': '大象', 'bear': '熊', 'zebra': '斑马',
    'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞',
    'handbag': '手提包', 'tie': '领带', 'suitcase': '行李箱',
    'frisbee': '飞盘', 'skis': '滑雪板', 'snowboard': '滑雪板',
    'sports ball': '球', 'kite': '风筝', 'baseball bat': '棒球棒',
    'baseball glove': '棒球手套', 'skateboard': '滑板', 'surfboard': '冲浪板',
    'tennis racket': '网球拍', 'bottle': '瓶子', 'wine glass': '酒杯',
    'cup': '杯子', 'fork': '叉子', 'knife': '刀', 'spoon': '勺子',
    'bowl': '碗', 'banana': '香蕉', 'apple': '苹果', 'sandwich': '三明治',
    'orange': '橙子', 'broccoli': '西兰花', 'carrot': '胡萝卜',
    'hot dog': '热狗', 'pizza': '披萨', 'donut': '甜甜圈', 'cake': '蛋糕',
    'chair': '椅子', 'couch': '沙发', 'potted plant': '盆栽',
    'bed': '床', 'dining table': '餐桌', 'toilet': '马桶', 'tv': '电视',
    'laptop': '笔记本电脑', 'mouse': '鼠标', 'remote': '遥控器',
    'keyboard': '键盘', 'cell phone': '手机', 'microwave': '微波炉',
    'oven': '烤箱', 'toaster': '烤面包机', 'sink': '水槽',
    'refrigerator': '冰箱', 'book': '书', 'clock': '钟', 'vase': '花瓶',
    'scissors': '剪刀', 'teddy bear': '泰迪熊', 'hair drier': '吹风机',
    'toothbrush': '牙刷'
}

# ============ 提示词模板 ============

PROMPT_TEMPLATE = """你是一个图像描述专家。根据计算机视觉模型的分析结果，生成准确且多样化的图像描述。

## 图像分析结果
- 检测到的物体：{objects}
- 各物体数量：{counts}
- 位置分布：{positions}
- 场景类型：{scene}

1. **核对检测结果**：首先检查 YOLO 提供的检测信息，确认图像中物体及数量是否与实际一致。注意：如果你观察到图像与 YOLO 检测结果不符，请以你看到的为准，并指出差异。
2. **生成描述**：
   - 基于你观察到的图像内容，以及 YOLO 检测信息，生成 {num_candidates} 个不同的中文描述。
   - 每条描述长度在 {min_length}-{max_length} 字之间。
   - 保持描述自然流畅，像人类书写。
   - 在确保事实准确的前提下，一定要尽可能多地使用 YOLO 检测结果，如数量、位置信息。
   - 描述开头请明确人物或主体，避免一开始就使用“她”或“他”。
   - 描述要有变化（不同视角、不同重点、不同表达方式）。
   - 每条描述保持完整，两条描述之间不要有上下承接的关系。
   - 按可能性从高到低排列。

## 输出格式
直接输出 {num_candidates} 个描述，每行一个，不要编号，不要其他说明。

## 示例参考
输入：物体=['人', '椅子', '桌子'], 数量={{'人': 2, '椅子': 4, '桌子': 1}}, 场景='室内'
输出：
两个人坐在餐桌旁边交谈
室内场景中两人围坐在桌子边
四把椅子围绕着餐桌，两人正在使用其中两把
一间房间里有两个人和一张桌子
两人在摆放着四把椅子的桌子旁

现在开始生成："""

PROMPT_TEMPLATE_BASELINE = """你是一个图像描述专家。根据你看到的图片，生成一条准确的图像描述。

生成描述的要求：
- 描述长度在 {min_length}-{max_length} 字之间。
- 保持描述自然流畅，像人类书写。
- 描述开头请明确人物或主体，避免一开始就使用“她”或“他”。

## 输出格式
直接输出描述，不要其他说明。

现在开始生成："""

# ============ 场景分类 ============

INDOOR_OBJECTS = {'chair', 'couch', 'potted plant', 'bed', 'dining table', 
                  'toilet', 'tv', 'laptop', 'microwave', 'oven', 'refrigerator'}
OUTDOOR_OBJECTS = {'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                   'traffic light', 'bench', 'bird'}

def classify_scene(objects):
    """根据检测到的物体推断场景类型"""
    indoor_count = sum(1 for obj in objects if obj in INDOOR_OBJECTS)
    outdoor_count = sum(1 for obj in objects if obj in OUTDOOR_OBJECTS)
    
    if indoor_count > outdoor_count:
        return "室内"
    elif outdoor_count > indoor_count:
        return "室外"
    else:
        return "未知"

# ============ 路径配置 ============

# 输出目录
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日志配置
LOG_LEVEL = "INFO"
