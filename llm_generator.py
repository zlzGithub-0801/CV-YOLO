"""
LLM 候选生成模块
功能：基于 YOLO 结果，用 Qwen 生成多个候选描述
支持本地模型和API调用两种方式
"""

import config
import re
import base64


class LLMGenerator:
    """LLM候选描述生成器（支持本地模型和API调用）"""
    
    def __init__(self):
        """初始化LLM生成器"""
        self.use_api = config.LLM_USE_API
        
        if self.use_api:
            self._init_api()
        else:
            self._init_local_model()
    
    def _init_api(self):
        """初始化API调用"""
        print(f"[LLM] 使用 API 调用模式")
        print(f"[LLM] API 类型: {config.LLM_API_TYPE}")
        
        if config.LLM_API_TYPE == "dashscope":
            try:
                import dashscope
                self.dashscope = dashscope
                dashscope.api_key = config.DASHSCOPE_API_KEY
                print(f"[LLM] 阿里云通义千问 API 已配置")
                print(f"[LLM] 模型: {config.DASHSCOPE_MODEL}")
            except ImportError:
                raise ImportError("请安装 dashscope: pip install dashscope")
            
        elif config.LLM_API_TYPE == "openai":
            try:
                from openai import OpenAI
                # 使用新版 OpenAI SDK
                self.openai_client = OpenAI(
                    api_key=config.OPENAI_API_KEY,
                    base_url=config.OPENAI_API_BASE
                )
                print(f"[LLM] OpenAI 兼容 API 已配置")
                print(f"[LLM] API Base: {config.OPENAI_API_BASE}")
                print(f"[LLM] 模型: {config.OPENAI_MODEL}")
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        
        print(f"[LLM] API 初始化完成")
    
    def _init_local_model(self):
        """初始化本地模型"""
        print(f"[LLM] 使用本地模型模式")
        print(f"[LLM] 正在加载模型: {config.LLM_MODEL_NAME}")
        print(f"[LLM] 设备: {config.LLM_DEVICE}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            device = config.LLM_DEVICE
            if device == "cuda" and not torch.cuda.is_available():
                print("[LLM] 警告: 未检测到GPU，切换到CPU模式")
                device = "cpu"
            
            self.device = device
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL_NAME, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_NAME,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).eval()
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            print(f"[LLM] 本地模型加载完成")
            
        except ImportError:
            raise ImportError("请安装 transformers 和 torch: pip install transformers torch")
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_candidates(self, yolo_results, image_path, num_candidates=config.NUM_CANDIDATES):
        """
        生成候选描述
        
        Args:
            yolo_results: YOLO检测结果字典
            num_candidates: 候选描述数量
            
        Returns:
            list: 候选描述列表
        """
        print(f"[LLM] 正在生成 {num_candidates} 个候选描述...")
        
        # 构建提示词
        prompt = self._build_prompt(yolo_results, num_candidates)
        
        print(f"[LLM] 提示词已构建")
        
        # 根据模式调用不同的生成方法
        if self.use_api:
            response = self._generate_api(prompt, image_path)
        else:
            response = self._generate_local(prompt)
        
        print(f"[LLM] 原始输出:\n{response[:200]}...")
        print("-" * 50)
        
        # 解析候选描述
        candidates = self._parse_response(response, num_candidates)
        
        print(f"[LLM] 成功生成 {len(candidates)} 个候选")
        for i, cand in enumerate(candidates, 1):
            print(f"       {i}. {cand}")
        
        return candidates
    
    def _generate_api(self, prompt, image_path):
        """使用API生成文本"""
        if config.LLM_API_TYPE == "dashscope":
            return self._generate_dashscope(prompt)
        elif config.LLM_API_TYPE == "openai":
            return self._generate_openai(prompt, image_path)
        else:
            raise ValueError(f"不支持的API类型: {config.LLM_API_TYPE}")
    
    def _generate_dashscope(self, prompt):
        """使用阿里云通义千问API生成"""
        from dashscope import Generation
        
        response = Generation.call(
            model=config.DASHSCOPE_MODEL,
            prompt=prompt,
            max_tokens=config.LLM_MAX_LENGTH,
            temperature=config.LLM_TEMPERATURE,
            top_p=config.LLM_TOP_P,
        )
        
        if response.status_code == 200:
            return response.output.text
        else:
            raise Exception(f"API调用失败: {response.message}")
    
    def _generate_openai(self, prompt, image_path):
        """使用OpenAI兼容API生成（新版SDK）"""
        img_type = f"image/{image_path.split('.')[-1]}"
        img_b64_str = self.encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img_type};base64,{img_b64_str}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        completion = self.openai_client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            max_tokens=config.LLM_MAX_LENGTH,
            temperature=config.LLM_TEMPERATURE,
            top_p=config.LLM_TOP_P,
        )
        
        return completion.choices[0].message.content
    
    def _generate_local(self, prompt):
        """使用本地模型生成"""
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=None,
            max_length=config.LLM_MAX_LENGTH,
            temperature=config.LLM_TEMPERATURE,
            top_p=config.LLM_TOP_P,
            top_k=config.LLM_TOP_K
        )
        return response
    
    def _build_prompt(self, yolo_results, num_candidates):
        """构建提示词"""
        objects = yolo_results['objects']
        counts = yolo_results['counts']
        positions = yolo_results['positions']
        scene = yolo_results['scene']
        
        # 格式化位置信息
        position_summary = {obj: ', '.join(pos) if pos else "未知" 
                           for obj, pos in positions.items()}
        
        # 使用配置中的模板
        prompt = config.PROMPT_TEMPLATE.format(
            objects=', '.join(objects),
            counts=str(counts),
            positions=str(position_summary),
            scene=scene,
            num_candidates=num_candidates,
            min_length=config.MIN_CAPTION_LENGTH,
            max_length=config.MAX_CAPTION_LENGTH
        )
        
        return prompt
    
    def _parse_response(self, response, expected_num):
        """
        解析LLM输出，提取候选描述
        
        Args:
            response: LLM生成的文本
            expected_num: 期望的候选数量
            
        Returns:
            list: 候选描述列表
        """
        lines = response.strip().split('\n')
        
        candidates = []
        rejected_lines = []  # 记录被拒绝的行
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # 移除编号
            original_line = line
            line = re.sub(r'^\d+[.、\s]+', '', line)
            
            # 检查长度
            line_length = len(line)
            if config.MIN_CAPTION_LENGTH <= line_length <= config.MAX_CAPTION_LENGTH + 10:
                candidates.append(line)
                print(f"[LLM] ✓ 接受描述 (长度 {line_length}): {line[:50]}...")
            else:
                rejected_lines.append((line, line_length))
                print(f"[LLM] ✗ 拒绝描述 (长度 {line_length}, 要求 {config.MIN_CAPTION_LENGTH}-{config.MAX_CAPTION_LENGTH}): {line[:50]}...")
            
            if len(candidates) >= expected_num:
                break
        
        # 如果候选数量不足，显示诊断信息
        if len(candidates) < expected_num:
            print(f"\n[LLM] ⚠ 警告: 只生成了 {len(candidates)}/{expected_num} 个有效候选")
            
            if rejected_lines:
                print(f"[LLM] 共有 {len(rejected_lines)} 条描述因长度不符被拒绝:")
                for i, (text, length) in enumerate(rejected_lines[:5], 1):
                    print(f"       {i}. (长度 {length}) {text[:60]}...")
                
                # 如果没有有效候选但有被拒绝的候选，放宽限制
                if len(candidates) == 0 and rejected_lines:
                    print(f"\n[LLM] 尝试放宽长度限制以避免零候选...")
                    for text, length in rejected_lines[:expected_num]:
                        if length >= 20:  # 至少要有20个字
                            candidates.append(text)
                            print(f"[LLM] ✓ 降低标准后接受 (长度 {length}): {text[:50]}...")
        
        return candidates



# ============ 测试代码 ============

if __name__ == "__main__":
    mock_yolo_result = {
        'objects': ['人', '椅子', '桌子'],
        'counts': {'人': 2, '椅子': 4, '桌子': 1},
        'positions': {
            '人': ['画面中央', '画面右侧'],
            '椅子': ['画面左侧', '画面右侧', '画面中央', '画面中央'],
            '桌子': ['画面中央']
        },
        'scene': '室内'
    }
    
    try:
        generator = LLMGenerator()
        candidates = generator.generate_candidates(mock_yolo_result, num_candidates=10)
        
        print("\n" + "="*50)
        print("生成的候选描述:")
        for i, cand in enumerate(candidates, 1):
            print(f"{i}. {cand}")
        print("="*50)
        
    except Exception as e:
        print(f"[错误] {e}")
        import traceback
        traceback.print_exc()
