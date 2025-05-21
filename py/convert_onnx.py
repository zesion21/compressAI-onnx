from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from compressai.entropy_models.entropy_models import EntropyModel
from compressai.zoo import bmshj2018_factorized
from torch import nn
import time

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
net = bmshj2018_factorized(quality=8, pretrained=True).to(device).eval()


# 定义子模型，仅包含编码和量化
class EncoderWithEntropyBottleneck(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.g_a = model.g_a  # 编码器
        self.entropy_bottleneck = model.entropy_bottleneck  # EntropyBottleneck
        # self.compress = model.compress

    def forward(self, x):
        x_out = self.g_a(x)  # 编码器输出潜在表示
        # z_hat, likelihoods = self.entropy_bottleneck(x_out)  # 量化并估计概率
        # return z_hat, likelihoods
        indexes = self.entropy_bottleneck._build_indexes(x_out.size())
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x_out.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x_out.size(0), *([-1] * (spatial_dims + 1)))
        return x_out, indexes, medians


def convert(model: torch.nn.Module):
    inputPath = "E:/work2/c/engine_runtime/1.jpg"
    outPath = "examples/assets/1.bin"

    # 创建子模型
    encoder_model = EncoderWithEntropyBottleneck(model).eval()
    encoder_model = encoder_model.float()

    # 准备数据
    img = Image.open(inputPath).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device).float()
    # x = torch.randn(1, 3, 1024, 1024).to(device).float()
    print(x.shape)

    # 开始推理
    with torch.no_grad():
        pytorch_outputs = encoder_model(x)
        x_out_pytorch, indexes_pytorch, medians_pytorch = pytorch_outputs
        # s = encoder_model.compress(x)["strings"][0][0]
        # with open(outPath, "wb") as f:
        #     f.write(s)

        # print(medians_pytorch, x_out_pytorch)
    # 保存 PyTorch 输出
    torch.save(x_out_pytorch, "examples/test/x_out_pytorch.pt")
    torch.save(indexes_pytorch, "examples/test/indexes_pytorch.pt")
    torch.save(medians_pytorch, "examples/test/medians_pytorch.pt")

    print("x_out_pytorch shape:", x_out_pytorch.shape)
    print("indexes_pytorch shape:", indexes_pytorch.shape)
    print("medians_pytorch shape:", medians_pytorch.shape)

    # dummy_input = torch.randn(1, 3, 256, 256).to(device)
    torch.onnx.export(
        encoder_model,
        x,
        "examples/test/bmshj2018_factorized_encoder_self2.onnx",
        opset_version=13,
        input_names=["input"],
        output_names=["x_out", "indexes", "medians"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "x_out": {0: "batch_size", 2: "height_latent", 3: "width_latent"},
            "indexes": {
                0: "batch_size",
            },
            "medians": {0: "batch_size", 2: "height_latent", 3: "width_latent"},
        },
        # output_names=["z_hat", "likelihoods"],
        # dynamic_axes={
        #     "input": {0: "batch_size", 2: "height", 3: "width"},
        #     "z_hat": {0: "batch_size", 2: "height_latent", 3: "width_latent"},
        #     "likelihoods": {0: "batch_size", 2: "height_latent", 3: "width_latent"},
        # },
    )


if __name__ == "__main__":
    # checkSunModel(net)
    start = time.time()
    convert(net)
    end = time.time()

    print(f"程序运行完成，耗时：{(end-start):.4f}s")
    # checkOxxn()
    # name = "stmalo_fracape"
    # inputPath = f"examples/assets/{name}.png"
    # outPath = f"examples/assets/{name}_bin"
    # doPress(inputPath, outPath)

    # input = f"examples/assets/stmalo_fracape_bin_768_512.bin"
    # output = f"examples/assets/stmalo_fracape_bin_768_512.png"
    # doDepress(input, output)
