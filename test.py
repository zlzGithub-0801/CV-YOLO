# import time
# import config
# from openai import OpenAI

# import base64
 
# def encode_image(image_path):
#   with open(image_path, "rb") as image_file:
#     return base64.b64encode(image_file.read()).decode('utf-8')

# client = OpenAI(
#     api_key=config.OPENAI_API_KEY,
#     base_url=config.OPENAI_API_BASE
# )

# img_type = "image/jpg"
# image_path = "pizza.jpg"
# img_b64_str = encode_image(image_path)
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:{img_type};base64,{img_b64_str}"
#                 }
#             },
#             {
#                 "type": "text",
#                 "text": "描述这张图片，输出要在15-50字之间。"
#             }
#         ]
#     }
# ]

# start = time.time()
# response = client.chat.completions.create(
#     model=config.OPENAI_MODEL,
#     messages=messages,
#     max_tokens=2048
# )
# print(f"Response costs: {time.time() - start:.2f}s")
# print(f"Generated text: {response.choices[0].message.content}")

import json
with open("cvtest/info.json", "r", encoding="utf-8") as f:
    info_data = json.load(f)

print(info_data[0]["instances"][0].keys())
print(info_data[0]["captions"][0].keys())
print(info_data[0]["panoptic"].keys())