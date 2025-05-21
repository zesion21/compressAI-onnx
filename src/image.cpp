#include "../includes/image.h"

void greet(std::string name)
{

    std::cout << "Hello " << name << std::endl;
}

std::vector<float> imageToTensor(const std::string &image_path, int &height, int &width)
{
    // ����ͼƬ
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    // ��ȡ�ߴ�
    height = img.rows;
    width = img.cols;

    // ת��Ϊ RGB��OpenCV Ĭ�� BGR��
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // ��һ���� [0, 1]
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // �������� (1, 3, height, width)
    std::vector<float> tensor(1 * 3 * height * width);

    // ͨ������
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    // ���������R, G, B ˳��
    for (int c = 0; c < 3; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                float rounded = std::round(channels[c].at<float>(h, w) * 10000.0f) / 10000.0f;
                tensor[c * height * width + h * width + w] = rounded;
            }
        }
    }

    return tensor;
}