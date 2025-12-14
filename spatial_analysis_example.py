"""
改进建议：基于边界框的空间关系分析
"""

# 在 yolo_detector.py 的 detect() 方法中添加以下功能：

def analyze_spatial_relationships(self, boxes, class_names):
    """
    分析物体之间的空间关系
    
    Args:
        boxes: YOLO检测到的边界框列表
        class_names: 对应的类别名称列表
        
    Returns:
        dict: 空间关系描述
    """
    relationships = []
    
    # 找出所有"人"和关键物体（桌子、食物等）
    people_boxes = [(i, box) for i, (box, cls) in enumerate(zip(boxes, class_names)) if cls == 'person']
    table_boxes = [(i, box) for i, (box, cls) in enumerate(zip(boxes, class_names)) if cls in ['dining table', 'table']]
    food_boxes = [(i, box) for i, (box, cls) in enumerate(zip(boxes, class_names)) 
                  if cls in ['pizza', 'bowl', 'cup', 'fork', 'knife']]
    
    # 对每个人分析其与桌子/食物的关系
    for person_idx, person_box in people_boxes:
        person_area = calculate_box_area(person_box)
        person_center = get_box_center(person_box)
        
        # 计算与桌子的距离
        if table_boxes:
            table_idx, table_box = table_boxes[0]
            distance_to_table = calculate_distance(person_center, get_box_center(table_box))
            overlap = calculate_iou(person_box, table_box)  # 计算重叠度
            
            if overlap > 0.1 or distance_to_table < 100:  # 阈值可调
                relationships.append(f"人物{person_idx+1}靠近餐桌")
            else:
                relationships.append(f"人物{person_idx+1}远离餐桌")
        
        # 分析边界框大小（大框 = 近景，小框 = 远景）
        if person_area > 10000:  # 阈值可调
            relationships.append(f"人物{person_idx+1}在近处（前景）")
        else:
            relationships.append(f"人物{person_idx+1}在远处（背景）")
    
    return relationships


def calculate_box_area(box):
    """计算边界框面积"""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def get_box_center(box):
    """获取边界框中心点"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_distance(point1, point2):
    """计算两点距离"""
    import math
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_iou(box1, box2):
    """计算IoU（交并比）"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0
