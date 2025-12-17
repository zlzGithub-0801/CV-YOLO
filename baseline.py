import argparse
import base64
import config
from clip_ranker import CLIPRanker
import json
import os
import time
from openai import OpenAI

def is_image_file(filename):
    return filename.lower().endswith((
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
    ))

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def generate_image_description(image_path):
    client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE
    )

    img_type = "image/{}".format(image_path.split('.')[-1])
    img_b64_str = encode_image(image_path)

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
                    "text": config.PROMPT_TEMPLATE_BASELINE.format(
                        min_length=config.MIN_CAPTION_LENGTH,
                        max_length=config.MAX_CAPTION_LENGTH
                    )
                }
            ]
        }
    ]

    start = time.time()
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        max_tokens=2048
    )
    time_cost = time.time() - start
    generated_text = response.choices[0].message.content
    return time_cost, generated_text

def generate_and_save_results(image_path, output_json_path, cr, save_result):
    time_cost, generated_text = generate_image_description(image_path)
    clip_score = None
    if cr is not None:
        _, clip_score = cr.rank_captions(
            image_path,
            [generated_text]
        )[0]
    print(f"Response costs: {time_cost:.2f}s")
    print(f"Generated text: {generated_text}")
    print(f"CLIP score: {clip_score:.4f}")

    if not save_result:
        return

    data = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []

    data.append({
        "image_name": image_path.split('/')[-1],
        "generated_text": generated_text,
        "clip_score": float(clip_score),
        "time_cost": time_cost
    })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    """
    使用方式：python baseline.py [图片路径或文件夹路径] [--output_dir 输出目录] [--save_result]
    例子：python baseline.py cvtest/images --output_dir results --save_result
    说明：如果路径是文件夹，则会处理该文件夹下的所有图片文件。 
    --output_dir 应为一个目录，或不输入该参数。如果输入该参数，结果会保存到 output_dir/output_baseline.json 文件中。如果不输入该参数，结果会保存到 ./output_baseline.json 文件中。
    --save_result 可选，表示是否保存结果到文件，默认为保存。如果不想保存结果，可以添加 --no-save_result 参数。
    """
    parser = argparse.ArgumentParser(
        description="Generate image descriptions for an image or a folder of images"
    )
    parser.add_argument(
        "path",
        type=str,
        default="cvtest/images",   # 默认路径
        help="Path to an image file or a folder containing images (default: cvtest/images)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--save_result",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save output_baseline.json (default: True)"
    )
    args = parser.parse_args()

    input_path = args.path

    if args.output_dir is None:
        output_dir = "."
    else:
        output_dir = args.output_dir

    output_json_path = os.path.join(output_dir, "output_baseline.json")

    cr = CLIPRanker()
    if os.path.isfile(input_path):
        # 单张图片
        if not is_image_file(input_path):
            raise ValueError(f"Not an image file: {input_path}")
        generate_and_save_results(input_path, output_json_path, cr, args.save_result)

    elif os.path.isdir(input_path):
        for filename in sorted(os.listdir(input_path)):
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path) and is_image_file(file_path):
                generate_and_save_results(file_path, output_json_path, cr, args.save_result)

    else:
        raise ValueError(f"Invalid path: {input_path}")