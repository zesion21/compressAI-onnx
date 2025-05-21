import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


# 检查输出是否接近
def check_tensors_close(pytorch_tensor, onnx_tensor, name, rtol=1e-5, atol=1e-8):
    if pytorch_tensor.shape != onnx_tensor.shape:
        print(
            f"Shape mismatch for {name}: PyTorch {pytorch_tensor.shape}, ONNX {onnx_tensor.shape}"
        )
        return False
    close = torch.allclose(pytorch_tensor, onnx_tensor, rtol=rtol, atol=atol)
    if not close:
        max_diff = torch.max(torch.abs(pytorch_tensor - onnx_tensor))
        print(f"Max difference for {name}: {max_diff}")
    return close


# 加载 ONNX 模型
session = ort.InferenceSession(
    "examples/test/bmshj2018_factorized_encoder_self2.onnx",
    providers=["CPUExecutionProvider"],
)
inputPath = "E:/work2/c/engine_runtime/1.jpg"
# 准备输入（转换为 NumPy 数组）

img = Image.open(inputPath).convert("RGB")
x = transforms.ToTensor()(img).unsqueeze(0).float()
input_np = np.array(x).astype(np.float32)
# input_np = np.transpose(input_np, (2, 0, 1))
# input_np = np.expand_dims(input_np, axis=0)
print(input_np.shape)
input_feed = {"input": input_np}

# # 运行推理
onnx_outputs = session.run(None, input_feed)
x_out_onnx, indexes_onnx, medians_onnx = [torch.from_numpy(out) for out in onnx_outputs]

# # 打印 ONNX 输出形状
print("x_out_onnx shape:", x_out_onnx.shape)
print("indexes_onnx shape:", indexes_onnx.shape)
print("medians_onnx shape:", medians_onnx.shape)

x_out_pytorch = torch.load("examples/test/x_out_pytorch.pt", map_location="cpu")
indexes_pytorch = torch.load("examples/test/indexes_pytorch.pt", map_location="cpu")
medians_pytorch = torch.load("examples/test/medians_pytorch.pt", map_location="cpu")


# 比较所有输出
print("x_out close:", check_tensors_close(x_out_pytorch, x_out_onnx, "x_out"))
print("indexes close:", check_tensors_close(indexes_pytorch, indexes_onnx, "indexes"))
print("medians close:", check_tensors_close(medians_pytorch, medians_onnx, "medians"))
