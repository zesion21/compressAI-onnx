# compressAI-onnx

A c++ demo， loading onnx model which is exported by torch！

## 目录结构

```
—— project
|
|—— out
|
|—— py // 基于compressai ， 可直接放到compressai运行
|     entropy_bottleneck_extrc_cbf.py // 用于提取模型的cdf到json
|     MyDonnx.py //测试导出的onnx模型和原模型的差异
|     convert_onnx.py //用于将 模型转换为 onnx
|
|—— models
|       bmshj2018_factorized_encoder.onnx // 一个测试模型
|       entropy_cdf.json // 模型对用的cdf文件
|
|
```
