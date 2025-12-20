"""
工具函数模块
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体（根据操作系统调整）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_image(image_path):
    """
    加载图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        PIL.Image: 图像对象
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像不存在: {image_path}")
    
    return Image.open(image_path)


def visualize_results(image_path, yolo_result, candidates, ranked_captions, save_path=None):
    """
    可视化完整结果
    
    Args:
        image_path: 原始图像路径
        yolo_result: YOLO检测结果
        candidates: LLM生成的所有候选
        ranked_captions: CLIP排序后的结果 [(描述, 分数), ...]
        save_path: 保存路径（可选）
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 原始图像
    ax1 = plt.subplot(2, 3, 1)
    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.set_title("原始图像", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. YOLO检测结果
    ax2 = plt.subplot(2, 3, 2)
    annotated = yolo_result['raw_results'][0].plot()
    # OpenCV BGR to RGB
    annotated_rgb = annotated[:, :, ::-1]
    ax2.imshow(annotated_rgb)
    ax2.set_title("YOLO 检测结果", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. YOLO分析信息
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    info_text = f"""YOLO 分析结果
    
检测到的物体:
{', '.join(yolo_result['objects'])}

物体数量:
{yolo_result['counts']}

场景类型: {yolo_result['scene']}
"""
    ax3.text(0.1, 0.5, info_text, fontsize=11, 
             verticalalignment='center', family='monospace')
    ax3.set_title("场景分析", fontsize=14, fontweight='bold')
    
    # 4. LLM候选列表（前10个）
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    candidates_text = "LLM 生成的候选 (前10个):\n\n"
    for i, cand in enumerate(candidates[:10], 1):
        candidates_text += f"{i}. {cand}\n"
    ax4.text(0.05, 0.95, candidates_text, fontsize=9,
             verticalalignment='top', family='monospace')
    ax4.set_title("候选描述", fontsize=14, fontweight='bold')
    
    # 5. CLIP排序结果（前10个）
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    ranking_text = "CLIP 相似度排序 (Top 10):\n\n"
    for i, (caption, score) in enumerate(ranked_captions[:10], 1):
        ranking_text += f"{i}. [{score:.3f}] {caption}\n"
    ax5.text(0.05, 0.95, ranking_text, fontsize=9,
             verticalalignment='top', family='monospace',
             color='blue' if i == 1 else 'black')
    ax5.set_title("相似度排序", fontsize=14, fontweight='bold')
    
    # 6. 最终结果
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    best_caption, best_score = ranked_captions[0]
    final_text = f"""最终生成的图像描述:

"{best_caption}"

相似度分数: {best_score:.4f}

---
处理流程:
1. YOLO 检测物体
2. LLM 生成 {len(candidates)} 个候选
3. CLIP 计算相似度
4. 选择最佳描述
"""
    ax6.text(0.1, 0.5, final_text, fontsize=12,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax6.set_title("最终结果", fontsize=14, fontweight='bold', color='green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[可视化] 结果已保存: {save_path}")
    
    plt.show()


def save_results_to_file(image_path, yolo_result, candidates, ranked_captions, output_file):
    """
    保存结果到文本文件
    
    Args:
        image_path: 图像路径
        yolo_result: YOLO结果
        candidates: 候选列表
        ranked_captions: 排序后的候选
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("图像描述生成系统 - 结果报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"输入图像: {image_path}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("1. YOLO 检测结果\n")
        f.write("-"*60 + "\n")
        f.write(f"检测到的物体: {', '.join(yolo_result['objects'])}\n")
        f.write(f"物体数量: {yolo_result['counts']}\n")
        f.write(f"场景类型: {yolo_result['scene']}\n\n")
        
        f.write("-"*60 + "\n")
        f.write(f"2. LLM 生成的候选描述 ({len(candidates)} 个)\n")
        f.write("-"*60 + "\n")
        for i, cand in enumerate(candidates, 1):
            f.write(f"{i}. {cand}\n")
        f.write("\n")
        
        f.write("-"*60 + "\n")
        f.write("3. CLIP 相似度排序\n")
        f.write("-"*60 + "\n")
        for i, (caption, score) in enumerate(ranked_captions, 1):
            f.write(f"{i}. [{score:.4f}] {caption}\n")
        f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("最终结果\n")
        f.write("="*60 + "\n")
        best_caption, best_score = ranked_captions[0]
        f.write(f'描述: "{best_caption}"\n')
        f.write(f"相似度分数: {best_score:.4f}\n")
        f.write("="*60 + "\n")
    
    print(f"[保存] 结果已保存到: {output_file}")


def create_summary_comparison(ranked_captions, num_show=5):
    """
    创建候选对比图表
    
    Args:
        ranked_captions: 排序后的候选
        num_show: 显示数量
    """
    captions = [c for c, _ in ranked_captions[:num_show]]
    scores = [s for _, s in ranked_captions[:num_show]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(captions))
    bars = ax.barh(y_pos, scores, color=['green' if i == 0 else 'skyblue' 
                                          for i in range(len(scores))])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{i+1}. {c[:30]}..." if len(c) > 30 else f"{i+1}. {c}" 
                        for i, c in enumerate(captions)])
    ax.invert_yaxis()
    ax.set_xlabel('相似度分数', fontsize=12)
    ax.set_title('Top 候选描述相似度对比', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(scores) * 1.1)
    
    # 添加分数标签
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score, i, f' {score:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def is_image_file(filename):
    return filename.lower().endswith((
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
    ))


# ============ 测试代码 ============

if __name__ == "__main__":
    print("工具函数模块测试")
    
    # 测试图像加载
    test_path = "test.jpg"
    if os.path.exists(test_path):
        img = load_image(test_path)
        print(f"成功加载图像: {img.size}")
    else:
        print("未找到测试图像")
