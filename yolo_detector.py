"""
YOLO 物体检测模块
功能：检测图像中的物体、位置和场景信息
"""

from ultralytics import YOLO
from collections import Counter
import config


class YOLODetector:
    """YOLO物体检测器"""
    
    def __init__(self, model_name=config.YOLO_MODEL):
        """
        初始化YOLO模型
        
        Args:
            model_name: YOLO模型名称，如 'yolov8n.pt'
        """
        print(f"[YOLO] 正在加载模型: {model_name}")
        self.model = YOLO(model_name)
        print(f"[YOLO] 模型加载完成")
    
    def detect(self, image_path):
        """
        检测图像中的物体
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: {
                'objects': ['人', '椅子', '桌子'],  # 检测到的物体列表（去重）
                'counts': {'人': 2, '椅子': 1},    # 各物体数量
                'positions': {'人': ['画面中央', '画面右侧']},  # 物体位置
                'scene': '室内',  # 场景类型
                'raw_results': results  # 原始检测结果（用于可视化）
            }
        """
        print(f"[YOLO] 正在检测图像: {image_path}")
        
        # 执行检测
        results = self.model(
            image_path, 
            conf=config.YOLO_CONF_THRESHOLD,
            iou=config.YOLO_IOU_THRESHOLD,
            verbose=False
        )
        
        # 解析结果
        detections = results[0]  # 第一张图像的结果
        boxes = detections.boxes
        
        objects_en = []  # 英文物体名称列表
        objects_zh = []  # 中文物体名称列表
        positions = {}   # 物体位置字典
        
        # 获取图像尺寸（用于归一化坐标）
        img_height, img_width = detections.orig_shape
        
        # 遍历每个检测到的物体
        for box in boxes:
            # 获取类别
            class_id = int(box.cls[0])
            class_name_en = self.model.names[class_id]
            class_name_zh = config.YOLO_CLASS_NAMES_ZH.get(class_name_en, class_name_en)
            
            objects_en.append(class_name_en)
            objects_zh.append(class_name_zh)
            
            # 获取边界框中心坐标
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            x_center = ((xyxy[0] + xyxy[2]) / 2) / img_width
            y_center = ((xyxy[1] + xyxy[3]) / 2) / img_height
            
            # 计算位置描述
            position = config.get_position_description(x_center, y_center)
            
            # 记录位置
            if class_name_zh not in positions:
                positions[class_name_zh] = []
            positions[class_name_zh].append(position)
        
        # 统计物体数量
        counts = dict(Counter(objects_zh))
        
        # 去重物体列表
        unique_objects = list(set(objects_zh))
        
        # 推断场景类型
        scene = config.classify_scene(objects_en)
        
        result = {
            'objects': unique_objects,
            'counts': counts,
            'positions': positions,
            'scene': scene,
            'raw_results': results
        }
        
        print(f"[YOLO] 检测完成: 发现 {len(unique_objects)} 种物体")
        print(f"       物体: {unique_objects}")
        print(f"       数量: {counts}")
        print(f"       场景: {scene}")
        
        return result
    
    def visualize(self, detection_result, save_path=None, show=False):
        """
        可视化检测结果
        
        Args:
            detection_result: detect() 返回的结果字典
            save_path: 保存路径（可选）
            show: 是否显示图像
        """
        results = detection_result['raw_results']
        
        # 使用YOLO内置的可视化
        annotated = results[0].plot()
        
        if save_path:
            import cv2
            cv2.imwrite(save_path, annotated)
            print(f"[YOLO] 可视化结果已保存: {save_path}")
        
        if show:
            import cv2
            cv2.imshow("YOLO Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated


# ============ 测试代码 ============

if __name__ == "__main__":
    import sys
    
    # 使用示例
    detector = YOLODetector()
    
    # 测试图像路径
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        print("用法: python yolo_detector.py <图像路径>")
        print("将使用默认测试（需要提供测试图像）")
        sys.exit(1)
    
    # 执行检测
    result = detector.detect(test_image)
    
    # 打印结果
    print("\n" + "="*50)
    print("检测结果:")
    print(f"物体列表: {result['objects']}")
    print(f"物体数量: {result['counts']}")
    print(f"物体位置: {result['positions']}")
    print(f"场景类型: {result['scene']}")
    print("="*50)
    
    # 保存可视化结果
    detector.visualize(result, save_path="yolo_output.jpg", show=False)
