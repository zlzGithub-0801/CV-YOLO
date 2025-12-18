import json

# 文件路径
output_path = "total_output/output.json"
baseline_path = "total_output/output_baseline.json"

# 读取 JSON
with open(output_path, "r", encoding="utf-8") as f:
    output_data = json.load(f)

with open(baseline_path, "r", encoding="utf-8") as f:
    baseline_data = json.load(f)

# 建立映射：统一 image_name（去掉 .jpg 后缀）
def normalize_name(name):
    return name.split(".")[0]

output_map = {normalize_name(item["image_name"]): item for item in output_data}
baseline_map = {normalize_name(item["image_name"]): item for item in baseline_data}

# ---------- 找出 output 比 baseline 高的图片 ----------
better_images = []
total_output_score = 0.0
total_baseline_score = 0.0
total_output_time = 0.0
total_baseline_time = 0.0
count = 0
better_count = 0

for name in baseline_map:
    if name not in output_map:
        print(f"⚠️ 警告：pipeline 中缺少图片 {name} 的结果，已跳过该图片的比较。")
        continue
    output_item = output_map[name]
    baseline_item = baseline_map[name]

    output_score = float(output_item.get("clip_score", 0.0))
    baseline_score = float(baseline_item.get("clip_score", 0.0)) 
    total_output_score += output_score
    total_baseline_score += baseline_score
    total_output_time += float(output_item.get("time_cost", {}).get("total", 0.0))
    total_baseline_time += float(baseline_item.get("time_cost", 0.0))
    if output_score > baseline_score:
        better_images.append(name)
    count += 1
avg_output_score = total_output_score / count if count > 0 else 0.0
avg_baseline_score = total_baseline_score / count if count > 0 else 0.0
avg_output_time = total_output_time / count if count > 0 else 0.0
avg_baseline_time = total_baseline_time / count if count > 0 else 0

print(f"总检测图像数量: {len(output_map)}")
print(f"clip_score 高于 baseline 的数量: {len(better_images)}")
print(f"总体平均 clip_score - pipeline:   {avg_output_score:.4f}")
print(f"总体平均 clip_score - baseline: {avg_baseline_score:.4f}")
print(f"总体平均用时 - pipeline:   {avg_output_time:.2f} s")
print(f"总体平均用时 - baseline: {avg_baseline_time:.2f} s")

# ---------- 仅对 better_images 统计 ----------
total_output_score = 0.0
total_baseline_score = 0.0
total_output_time = 0.0
total_baseline_time = 0.0

for name in better_images:
    output_item = output_map[name]
    baseline_item = baseline_map[name]

    total_output_score += float(output_item.get("clip_score", 0.0))
    total_baseline_score += float(baseline_item.get("clip_score", 0.0))

    total_output_time += float(output_item.get("time_cost", {}).get("total", 0.0))
    total_baseline_time += float(baseline_item.get("time_cost", 0.0))

count = len(better_images)
avg_output_score = total_output_score / count if count > 0 else 0.0
avg_baseline_score = total_baseline_score / count if count > 0 else 0.0
avg_output_time = total_output_time / count if count > 0 else 0.0
avg_baseline_time = total_baseline_time / count if count > 0 else 0.0

print("\n===== Better Subset Statistics =====")
print(f"样本数量: {count}")
print(f"平均 clip_score - pipeline:   {avg_output_score:.4f}")
print(f"平均 clip_score - baseline: {avg_baseline_score:.4f}")
print(f"平均用时 - pipeline:   {avg_output_time:.2f} s")
print(f"平均用时 - baseline: {avg_baseline_time:.2f} s")
