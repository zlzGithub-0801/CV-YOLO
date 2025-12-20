"""
CLIP 图像-文本匹配模块
功能：计算候选描述与图像的相似度，进行排序
"""

import torch
from PIL import Image
import numpy as np
import config

# 根据配置选择CLIP模型
if config.CLIP_MODEL_TYPE == "chinese-clip":
    from cn_clip.clip import load_from_name, tokenize
    USE_CHINESE_CLIP = True
else:
    import clip
    USE_CHINESE_CLIP = False


class CLIPRanker:
    """CLIP图像-文本匹配排序器"""
    
    def __init__(self, model_name=config.CLIP_MODEL_NAME):
        """
        初始化CLIP模型
        
        Args:
            model_name: CLIP模型名称
        """
        print(f"[CLIP] 正在加载模型: {model_name}")
        print(f"[CLIP] 模型类型: {config.CLIP_MODEL_TYPE}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CLIP] 设备: {self.device}")
        
        if USE_CHINESE_CLIP:
            # 使用 Chinese-CLIP
            self.model, self.preprocess = load_from_name(
                model_name, 
                device=self.device,
                download_root='./model_cache'
            )
        else:
            # 使用 OpenAI CLIP
            self.model, self.preprocess = clip.load(
                model_name, 
                device=self.device
            )
        
        self.model.eval()
        print(f"[CLIP] 模型加载完成")
    
    def rank_captions(self, image_path, candidates):
        """
        计算相似度并排序候选描述
        
        Args:
            image_path: 图像文件路径
            candidates: 候选描述列表
            
        Returns:
            list: [(描述, 相似度分数), ...] 按分数降序排列
        """
        print(f"[CLIP] 正在计算 {len(candidates)} 个候选的相似度...")
        
        # 加载并预处理图像
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        
        # 对文本进行编码
        if USE_CHINESE_CLIP:
            text_tokens = tokenize(candidates).to(self.device)
        else:
            text_tokens = clip.tokenize(candidates, truncate=True).to(self.device)
        
        # 计算特征
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_tokens)
            
            # 归一化
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # 计算余弦相似度
            similarity = (image_features @ text_features.T).squeeze(0)
            similarity = similarity.cpu().numpy()
        
        # 组合结果并排序
        results = list(zip(candidates, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"[CLIP] 相似度计算完成")
        print(f"       最高分: {results[0][1]:.4f}")
        print(f"       最低分: {results[-1][1]:.4f}")
        
        return results
    
    def get_best_caption(self, image_path, candidates, top_k=1):
        """
        获取最佳描述
        
        Args:
            image_path: 图像路径
            candidates: 候选列表
            top_k: 返回前k个结果
            
        Returns:
            如果top_k=1，返回 (描述, 分数)
            否则返回 [(描述, 分数), ...] 列表
        """
        ranked = self.rank_captions(image_path, candidates)
        
        if top_k == 1:
            return ranked[0]
        else:
            return ranked[:top_k]


# ============ 测试代码 ============

if __name__ == "__main__":
    import sys
    
    # 测试候选
    test_candidates = [
        "两个人坐在餐桌旁边交谈",
        "室内场景中两人围坐在桌子边",
        "四把椅子围绕着餐桌，两人正在使用其中两把",
        "一间房间里有两个人和一张桌子",
        "两人在摆放着四把椅子的桌子旁",
        "一只猫在沙发上睡觉",  # 错误描述用于对比
        "一辆汽车停在路边",     # 错误描述用于对比
    ]
    
    # 测试图像路径
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        print("用法: python clip_ranker.py <图像路径>")
        print("将使用模拟候选进行测试")
        sys.exit(1)
    
    try:
        # 创建排序器
        ranker = CLIPRanker()
        
        # 计算相似度
        ranked_results = ranker.rank_captions(test_image, test_candidates)
        
        # 打印结果
        print("\n" + "="*50)
        print("排序结果:")
        for i, (caption, score) in enumerate(ranked_results, 1):
            print(f"{i}. [{score:.4f}] {caption}")
        print("="*50)
        
        # 获取最佳描述
        best_caption, best_score = ranker.get_best_caption(test_image, test_candidates)
        print(f"\n最佳描述: {best_caption}")
        print(f"相似度分数: {best_score:.4f}")
        
    except Exception as e:
        print(f"[错误] {e}")
        print("\n提示：请确保已安装 CLIP 模型")
        print("Chinese-CLIP: pip install cn-clip")
        print("OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git")
