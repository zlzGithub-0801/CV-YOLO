"""
图像描述生成系统 - 主程序
整合 YOLO + LLM + CLIP 实现基于 Socratic Models 的图像描述生成

使用方法:
    python 11.py <图像路径> [--num_candidates 20] [--visualize] [--save_result]

示例:
    python 11.py test.jpg --num_candidates 10 --visualize
"""

import argparse
import os
import time
from pathlib import Path

# 导入自定义模块
from yolo_detector import YOLODetector
from llm_generator import LLMGenerator
from clip_ranker import CLIPRanker
import utils
import config


class ImageCaptionGenerator:
    """图像描述生成系统"""
    
    def __init__(self):
        """初始化所有模块"""
        print("="*60)
        print("图像描述生成系统 - 基于 Socratic Models")
        print("="*60)
        print()
        
        # 初始化各模块
        self.yolo_detector = YOLODetector()
        print()
        
        self.llm_generator = LLMGenerator()
        print()
        
        self.clip_ranker = CLIPRanker()
        print()
        
        print("="*60)
        print("所有模块初始化完成！")
        print("="*60)
        print()
    
    def generate(self, image_path, num_candidates=config.NUM_CANDIDATES):
        """
        生成图像描述的完整流程
        
        Args:
            image_path: 图像路径
            num_candidates: 候选描述数量
            
        Returns:
            dict: {
                'best_caption': str,        # 最佳描述
                'best_score': float,        # 相似度分数
                'yolo_result': dict,        # YOLO结果
                'candidates': list,         # 所有候选
                'ranked_captions': list,    # 排序后的候选
                'time_cost': dict           # 各阶段耗时
            }
        """
        print(f"\n处理图像: {image_path}\n")
        
        time_cost = {}
        
        # ========== 步骤1: YOLO 检测 ==========
        print("▶ 步骤 1/3: YOLO 物体检测")
        t1 = time.time()
        yolo_result = self.yolo_detector.detect(image_path)
        time_cost['yolo'] = time.time() - t1
        print(f"   耗时: {time_cost['yolo']:.2f} 秒\n")
        
        # ========== 步骤2: LLM 生成候选 ==========
        print("▶ 步骤 2/3: LLM 生成候选描述")
        t2 = time.time()
        candidates = self.llm_generator.generate_candidates(
            yolo_result, 
            num_candidates=num_candidates
        )
        time_cost['llm'] = time.time() - t2
        print(f"   耗时: {time_cost['llm']:.2f} 秒\n")
        
        # 如果候选数量不足，警告
        if len(candidates) < num_candidates:
            print(f"   ⚠ 警告: 只生成了 {len(candidates)}/{num_candidates} 个候选\n")
        
        # ========== 步骤3: CLIP 排序 ==========
        print("▶ 步骤 3/3: CLIP 相似度计算与排序")
        t3 = time.time()
        ranked_captions = self.clip_ranker.rank_captions(image_path, candidates)
        time_cost['clip'] = time.time() - t3
        print(f"   耗时: {time_cost['clip']:.2f} 秒\n")
        
        # ========== 获取最佳结果 ==========
        best_caption, best_score = ranked_captions[0]
        
        # 总耗时
        time_cost['total'] = sum(time_cost.values())
        
        print("="*60)
        print("✓ 处理完成！")
        print("="*60)
        print(f"\n最终描述: \"{best_caption}\"")
        print(f"相似度分数: {best_score:.4f}")
        print(f"\n总耗时: {time_cost['total']:.2f} 秒")
        print(f"  - YOLO: {time_cost['yolo']:.2f}s")
        print(f"  - LLM:  {time_cost['llm']:.2f}s")
        print(f"  - CLIP: {time_cost['clip']:.2f}s")
        print("="*60)
        
        return {
            'best_caption': best_caption,
            'best_score': best_score,
            'yolo_result': yolo_result,
            'candidates': candidates,
            'ranked_captions': ranked_captions,
            'time_cost': time_cost
        }


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="图像描述生成系统 - YOLO + LLM + CLIP"
    )
    parser.add_argument(
        "image_path", 
        type=str, 
        help="输入图像路径"
    )
    parser.add_argument(
        "--num_candidates", 
        type=int, 
        default=config.NUM_CANDIDATES,
        help=f"候选描述数量 (默认: {config.NUM_CANDIDATES})"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="可视化结果"
    )
    parser.add_argument(
        "--save_result", 
        action="store_true",
        help="保存结果到文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.OUTPUT_DIR,
        help=f"输出目录 (默认: {config.OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    # 检查图像是否存在
    if not os.path.exists(args.image_path):
        print(f"错误: 图像不存在: {args.image_path}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建生成器
    try:
        generator = ImageCaptionGenerator()
    except Exception as e:
        print(f"\n错误: 初始化失败")
        print(f"详细信息: {e}")
        print("\n提示:")
        print("  1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("  2. 确保已下载必要的模型")
        return
    
    # 生成描述
    try:
        result = generator.generate(args.image_path, args.num_candidates)
    except Exception as e:
        print(f"\n错误: 生成失败")
        print(f"详细信息: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存结果
    if args.save_result:
        # 生成输出文件名
        image_name = Path(args.image_path).stem
        
        # 保存文本结果
        text_output = os.path.join(args.output_dir, f"{image_name}_result.txt")
        utils.save_results_to_file(
            args.image_path,
            result['yolo_result'],
            result['candidates'],
            result['ranked_captions'],
            text_output
        )
        
        # 保存YOLO可视化
        yolo_output = os.path.join(args.output_dir, f"{image_name}_yolo.jpg")
        generator.yolo_detector.visualize(result['yolo_result'], save_path=yolo_output)
    
    # 可视化
    if args.visualize:
        vis_output = None
        if args.save_result:
            vis_output = os.path.join(args.output_dir, f"{image_name}_visualization.png")
        
        utils.visualize_results(
            args.image_path,
            result['yolo_result'],
            result['candidates'],
            result['ranked_captions'],
            save_path=vis_output
        )


if __name__ == "__main__":
    main()
