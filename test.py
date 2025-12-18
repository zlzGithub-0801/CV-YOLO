# import os
# import random
# import shutil

# src_dir = "coco2017/val2017"
# dst_dir = "coco2017/testval2"

# os.makedirs(dst_dir, exist_ok=True)

# prefixes = ["0000000", "0000001", "0000002"]
# num_samples = 5
# selected_list = [
# "000000001353",
# "000000016502",
# "000000022589",
# "000000024567",
# "000000047769",
# "000000097022",
# "000000099114",
# "000000108440",
# "000000117719",
# "000000132329",
# "000000134112",
# "000000140583",
# "000000159977",
# "000000186938",
# "000000235857",
# "000000259097",
# "000000266082",
# "000000286708"
# ]
# all_files = os.listdir(src_dir)

# for prefix in prefixes:
#     candidates = [
#         f for f in all_files
#         if f.startswith(prefix) and f.lower().endswith(".jpg")
#     ]
#     available = list(set(candidates) - set(selected_list))

#     if len(available) < num_samples:
#         raise ValueError(f"{prefix} 可用图片不足 {num_samples} 张")

#     selected = random.sample(available, num_samples)

#     for fname in selected:
#         src_path = os.path.join(src_dir, fname)
#         dst_path = os.path.join(dst_dir, fname)
#         shutil.copy2(src_path, dst_path)

#     print(f"{prefix}: 已拷贝 {num_samples} 张")

# print("✅ 完成 testval 构建")


# import json
# import os

# output_path = "outputs/output.json"
# baseline_path = "outputs/output_baseline.json"

# def normalize_name(name: str):
#     """统一 image_name：去掉 .jpg 后缀"""
#     return os.path.splitext(name)[0]

# # 读取 json
# with open(output_path, "r", encoding="utf-8") as f:
#     output_data = json.load(f)

# with open(baseline_path, "r", encoding="utf-8") as f:
#     baseline_data = json.load(f)

# # 建立 baseline 的映射
# baseline_map = {
#     normalize_name(item["image_name"]): float(item["clip_score"])
#     for item in baseline_data
# }

# # 找出 clip_score 更高的项
# better_images = []

# for item in output_data:
#     name = normalize_name(item["image_name"])
#     output_score = float(item["clip_score"])

#     if name not in baseline_map:
#         continue  # baseline 里没有就跳过

#     baseline_score = baseline_map[name]

#     if output_score > baseline_score:
#         better_images.append(name)

# # 输出结果
# print(f"📈 clip_score 高于 baseline 的图片数量: {len(better_images)}")
# print("图片列表:")
# for name in better_images:
#     print(name)

import os
import shutil

src_dir = "coco2017/val2017"
dst_dir = "testimg"

os.makedirs(dst_dir, exist_ok=True)

selected_list = [
"000000001353",
"000000016502",
"000000022589",
"000000024567",
"000000047769",
"000000097022",
"000000099114",
"000000108440",
"000000117719",
"000000132329",
"000000134112",
"000000140583",
"000000159977",
"000000186938",
"000000235857",
"000000259097",
"000000266082",
"000000286708",
"000000017627",
"000000046252",
"000000136355",
"000000177383",
"000000206027",
"000000288685",
"000000002261",
"000000016598",
"000000061584",
"000000121673",
"000000153229",
"000000167572",
"000000031248",
"000000031296",
"000000031322",
"000000032610",
"000000038829",
"000000039484",
"000000041990",
"000000043435",
"000000044590",
"000000045596",
"000000049060",
"000000049759",
"000000050380",
"000000050679",
"000000051309",
"000000053626",
"000000054123",
"000000054593",
"000000055150",
"000000055167",
"000000057150",
"000000057597",
"000000057672",
"000000059044",
"000000059386",
"000000060507",
"000000063047",
"000000066038",
"000000066231",
"000000067616"
]
selected_list = list(set(selected_list))
missing = []
cnt = -1
for img_id in selected_list:
    cnt += 1
    fname = f"{img_id}.jpg"
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    if not os.path.exists(src_path):
        print(f"⚠️ 未找到文件: {fname}, {cnt}")
        missing.append(fname)
        continue

    shutil.copy2(src_path, dst_path)
    print(f"✅ 已复制: {fname}")

print("\n=== 复制完成 ===")
print(f"成功复制: {len(selected_list) - len(missing)} 张")
if missing:
    print("缺失文件列表:")
    for f in missing:
        print("  ", f)