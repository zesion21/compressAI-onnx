# export_cdf.py
import torch
import json
from compressai.zoo import bmshj2018_factorized  # 可替换为你的模型

# 加载模型
model = bmshj2018_factorized(quality=8, pretrained=True).eval().cpu()

# 获取 EntropyBottleneck 模型
entropy_bottleneck = model.entropy_bottleneck

# 导出 CDF 表
cdf_data = {
    "cdf": entropy_bottleneck._quantized_cdf.tolist(),  # [channels, length]
    "cdf_lengths": entropy_bottleneck._cdf_length.tolist(),
    "offset": entropy_bottleneck._offset.tolist(),
}

with open("entropy_cdf.json", "w") as f:
    json.dump(cdf_data, f, indent=2)

print("CDF table exported to entropy_cdf.json")
