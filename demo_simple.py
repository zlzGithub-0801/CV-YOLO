"""
简化版演示脚本 - 用于测试和演示

这个脚本提供了一个简化的接口，可以快速测试系统
不需要复杂的命令行参数
"""

import os
from pathlib import Path

# 导入核心模块
from yolo_detector import YOLODetector
from llm_generator import LLMGenerator
from clip_ranker import CLIPRanker
import utils


def demo_simple(image_path, num_candidates=10):
    """
    简化演示函数
    
    Args:
        image_path: 图像路径
        num_candidates: 候选数量
    """
    print("\n" + "="*70)
    print(" 🎨 图像描述生成系统 - 简化演示")
    print("="*70 + "\n")
    
    # 1. 初始化模型
    print("📦 正在加载模型...\n")
    yolo = YOLODetector()
    print()
    # 注意：LLM和CLIP的初始化可能需要较长时间
    print("⏳ 正在加载 LLM（这可能需要几分钟）...")
    llm = LLMGenerator()
    print()
    clip = CLIPRanker()
    print()
    
    print("✅ 所有模型加载完成！\n")
    print("="*70 + "\n")
    
    # 2. YOLO 检测
    print("🔍 步骤 1: YOLO 物体检测")
    print("-"*70)
    yolo_result = yolo.detect(image_path)
    
    print("\n检测结果:")
    print(f"  - 物体: {', '.join(yolo_result['objects'])}")
    print(f"  - 数量: {yolo_result['counts']}")
    print(f"  - 场景: {yolo_result['scene']}")
    print()
    
    # 3. LLM 生成
    print("\n💭 步骤 2: LLM 生成候选描述")
    print("-"*70)
    candidates = llm.generate_candidates(yolo_result, num_candidates=num_candidates)
    
    print(f"\n生成了 {len(candidates)} 个候选:")
    for i, cand in enumerate(candidates[:5], 1):  # 只显示前5个
        print(f"  {i}. {cand}")
    if len(candidates) > 5:
        print(f"  ... (还有 {len(candidates)-5} 个)")
    print()
    
    # 4. CLIP 排序
    print("\n🎯 步骤 3: CLIP 相似度排序")
    print("-"*70)
    ranked = clip.rank_captions(image_path, candidates)
    
    print("\nTop 5 候选:")
    for i, (caption, score) in enumerate(ranked[:5], 1):
        print(f"  {i}. [{score:.4f}] {caption}")
    print()
    
    # 5. 最终结果
    best_caption, best_score = ranked[0]
    
    print("\n" + "="*70)
    print(" 🏆 最终结果")
    print("="*70)
    print(f"\n描述: \"{best_caption}\"")
    print(f"相似度分数: {best_score:.4f}")
    print("\n" + "="*70 + "\n")
    
    # 6. 保存结果（可选）
    save = input("是否保存结果？(y/n): ").strip().lower()
    if save == 'y':
        output_dir = "demo_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        image_name = Path(image_path).stem
        
        # 保存文本结果
        text_file = os.path.join(output_dir, f"{image_name}_result.txt")
        utils.save_results_to_file(
            image_path, yolo_result, candidates, ranked, text_file
        )
        
        # 保存可视化
        vis_file = os.path.join(output_dir, f"{image_name}_visualization.png")
        utils.visualize_results(
            image_path, yolo_result, candidates, ranked, save_path=vis_file
        )
        
        print(f"\n✅ 结果已保存到 {output_dir}/ 目录")
    
    return {
        'yolo_result': yolo_result,
        'candidates': candidates,
        'ranked': ranked,
        'best_caption': best_caption,
        'best_score': best_score
    }


if __name__ == "__main__":
    import sys
    
    # 获取图像路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("请输入图像路径:")
        image_path = input("> ").strip()
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"\n❌ 错误: 找不到图像文件: {image_path}")
        sys.exit(1)
    
    # 运行演示
    try:
        result = demo_simple(image_path, num_candidates=10)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 提示:")
        print("  1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("  2. 确保模型文件已下载")
        print("  3. 如果内存不足，尝试减少候选数量")
