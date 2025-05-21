#include <algorithm> // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include "fstream"
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "includes/image.h"
#include "includes/encode.h"
#include "includes/rans_interface.hpp"

int loadImage(std::string inputPath, cv::Mat &blob, int &height, int &width)
{
    cv::Mat image = cv::imread(inputPath);
    if (image.empty())
    {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    height = image.rows;
    width = image.cols;

    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(width, height), cv::Scalar(0, 0, 0), true, false);
    return 0;
}

int save2csv(int total_elements, auto output_data, std::string name)
{
    std::ofstream output_file("E:/work2/c/oxxn_runtime_demo/output_" + name + ".csv"); // 输出文件路径
    if (!output_file.is_open())
    {
        std::cerr << "Failed to open output file!" << std::endl;
        return -1;
    }
    for (size_t i = 0; i < total_elements; ++i)
    {
        output_file << output_data[i] << "\n";
    }
    output_file.close();
    return 0;
}

int writeFile(std::string data, std::string outPath)
{
    std::ofstream outFile(outPath, std::ios::binary);

    if (!outFile)
    {
        std::cerr << "Error opening file for writing!" << std::endl;
        return 1;
    }

    size_t length = data.size();
    outFile.write(data.c_str(), length);

    if (!outFile)
    {
        std::cerr << "Error writing to file!" << std::endl;
        return 1;
    }

    // 关闭文件
    outFile.close();
    std::cout << "String written to binary file successfully!" << std::endl;
    return 0;
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    try
    {
        // 初始化 ONNX Runtime 环境
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");
        Ort::SessionOptions session_options;

        // 启用 CUDA 执行提供者
        // OrtCUDAProviderOptions cuda_options;
        // cuda_options.device_id = 0; // 使用第一个 GPU 设备
        // session_options.AppendExecutionProvider_CUDA(cuda_options);

        // 设置优化选项
        // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // session_options.SetIntraOpNumThreads(4); // CPU 线程数（GPU 推理也会用到 CPU）

        // 加载模型
        const ORTCHAR_T *modelPath = L"../models/bmshj2018_factorized_encoder.onnx";
        std::string inputPath = "E:/work2/pythonProjects/CompressAI-master/examples/assets/stmalo_fracape.png";
        std::string outPath = "E:/work2/pythonProjects/CompressAI-master/examples/assets/onnx_stmalo_fracape_768_512.bin";
        Ort::Session session(env, modelPath, session_options);

        int imageHeight = 0;
        int imageWidth = 0;

        cv::Mat blob;
        loadImage(inputPath, blob, imageHeight, imageWidth);

        // 创建输入张量
        std::vector<int64_t> input_shape = {1, 3, imageHeight, imageWidth};                  // 模型输入形状
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); // CPU 内存用于输入准备
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());

        // 设置模型输入和输出名称
        std::vector<const char *> input_names = {"input"};
        std::vector<const char *> output_names = {"x_out", "indexes", "medians"};

        // 运行推理
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 3);

        // 承接运行结果
        int latent_h = imageHeight / 16;
        int latent_w = imageWidth / 16;

        std::vector<float> x_out(1 * 320 * latent_w * latent_h);
        std::vector<int32_t> indexes(1 * 320 * latent_w * latent_h);
        std::vector<float> medians(1 * 320 * 1 * 1);

        for (size_t i = 0; i < output_tensors.size(); i++)
        {
            auto output_shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "Output shape: ";
            int total_elements = 1;
            for (auto m = 0; m < output_shape.size(); m++)
            {
                auto dim = output_shape[m];
                std::cout << dim << " ";
                total_elements = total_elements * dim;
                if (i == 0 && m == 2)
                    latent_h = dim;
                if (i == 0 && m == 3)
                    latent_w = dim;
            }
            std::cout << std::endl;

            if (output_names[i] == "indexes")
            {
                int32_t *output_data = output_tensors[i].GetTensorMutableData<int32_t>();
                // save2csv(total_elements, output_data, output_names[i]);
                indexes.resize(total_elements);
                for (size_t i = 0; i < total_elements; ++i)
                {
                    indexes[i] = output_data[i];
                }
            }
            else if (output_names[i] == "x_out")
            {
                float *output_data = output_tensors[i].GetTensorMutableData<float>();
                // save2csv(total_elements, output_data, output_names[i]);
                x_out.resize(total_elements);
                for (size_t i = 0; i < total_elements; ++i)
                {
                    x_out[i] = output_data[i];
                }
            }
            else
            {
                float *output_data = output_tensors[i].GetTensorMutableData<float>();
                // save2csv(total_elements, output_data, output_names[i]);
                medians.resize(total_elements);
                for (size_t i = 0; i < total_elements; ++i)
                {
                    medians[i] = output_data[i];
                }
            }
        }

        std::cout << "latent_h:" << latent_h << "  latent_w:" << latent_w << std::endl;
        // std::cout << "x_out size:" << x_out.size() << ", indexes size:" << indexes.size() << ", medians size:" << medians.size() << std::endl;

        std::vector<int32_t> out_put(1 * 320 * latent_w * latent_h);
        // 把输出x_out序列化到 out_put
        quantize_by_block(x_out, medians, out_put, 1, 320, latent_h, latent_w);
        std::cout << "quantize_by_block run over !" << std::endl;

        // 读取模型中提取的cdf
        std::vector<std::vector<int32_t>> cdfs;
        std::vector<int32_t> cdf_lengths;
        std::vector<int32_t> offsets;
        readJson(cdfs, cdf_lengths, offsets);
        std::cout << "read json over !" << std::endl;

        // std::cout << " out_put:" << out_put.size() << ", cdfs:" << cdfs.size()
        //           << ", cdf_lengths:" << cdf_lengths.size() << ", offsets:" << offsets.size() << std::endl;

        // 进行熵编码
        RansEncoder encoder;
        std::string str = encoder.encode_with_indexes(out_put, indexes, cdfs, cdf_lengths, offsets);

        // save2csv(out_put.size(), out_put, "out_put"); // 保存到csv，方便进行查看

        std::cout << "encode over" << std::endl;

        writeFile(str, outPath);
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Standard Error: " << e.what() << std::endl;
        return -1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "exit with message success! cast: " << duration.count() << " ms" << std::endl;

    return 0;
}
