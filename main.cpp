#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cv.hpp>
#include "MoGDistribution.h"

int main()
{
    auto img = cv::imread("banana.png", cv::IMREAD_COLOR);
    img.convertTo(img, CV_64FC3);
    img /= 255;
    cv::Rect bbForeground(200, 280, 270, 100);
    cv::Rect bbBackground(200, 50, 270, 100);

    using Distribution = MoGDistribution<5, 3>;
    auto makeDataPoint = [](cv::Vec3d const& d)
    {
        Distribution::Data data;
        data(0) = d[0];
        data(1) = d[1];
        data(2) = d[2];
        return data;
    };

    // Data for foreground
    std::vector<Distribution::Data> fgData(bbForeground.area());
    for (unsigned int y = 0; y < bbForeground.height; y++)
    {
        for (unsigned int x = 0; x < bbForeground.width; x++)
        {
            auto color = img.at<cv::Vec3d>(y + bbForeground.y, x + bbForeground.x);
            fgData[x + y * bbForeground.width] = makeDataPoint(color);
        }
    }
    // Train foreground
    Distribution foreground;
    if (!foreground.load("foreground.gmm"))
    {
        foreground.train(fgData.begin(), fgData.end(), 5e2);
        foreground.save("foreground.gmm");
    }
    std::cout << "=== Foreground ===" << std::endl;
    foreground.print();

    // Data for background
    std::vector<Distribution::Data> bgData(bbBackground.area());
    for (unsigned int y = 0; y < bbBackground.height; y++)
    {
        for (unsigned int x = 0; x < bbBackground.width; x++)
        {
            auto color = img.at<cv::Vec3d>(y + bbBackground.y, x + bbBackground.x);
            bgData[x + y * bbBackground.width] = makeDataPoint(color);
        }
    }
    // Train background
    Distribution background;
    if (!background.load("background.gmm"))
    {
        background.train(bgData.begin(), bgData.end(), 5e2);
        background.save("background.gmm");
    }
    std::cout << "=== Background ===" << std::endl;
    background.print();

    // Show foreground confidence image
    cv::Mat foregroundConf(img.rows, img.cols, CV_64FC1);
    for (unsigned int y = 0; y < img.rows; y++)
    {
        for (unsigned int x = 0; x < img.cols; x++)
            foregroundConf.at<double>(y, x) = foreground.confidence(makeDataPoint(img.at<cv::Vec3d>(y, x)));
    }
    cv::normalize(foregroundConf, foregroundConf, 0, 255, CV_MINMAX);
    cv::imshow("Foreground confidences", foregroundConf);

    // Show background confidence image
    cv::Mat backgroundConf(img.rows, img.cols, CV_64FC1);
    for (unsigned int y = 0; y < img.rows; y++)
    {
        for (unsigned int x = 0; x < img.cols; x++)
            backgroundConf.at<double>(y, x) = background.confidence(makeDataPoint(img.at<cv::Vec3d>(y, x)));
    }
    cv::normalize(backgroundConf, backgroundConf, 0, 255, CV_MINMAX);
    cv::imshow("Background confidences", backgroundConf);

    // Show foreground-background segmentation
    cv::Mat seg(img.rows, img.cols, CV_8UC3);
    for (unsigned int y = 0; y < img.rows; y++)
    {
        for (unsigned int x = 0; x < img.cols; x++)
        {
            if(backgroundConf.at<double>(y, x) > foregroundConf.at<double>(y, x))
                seg.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            else
                seg.at<cv::Vec3b>(y, x) = cv::Vec3b(95, 218, 246);
        }
    }
    cv::imshow("Segmentation", seg);

    // Show original image with boxes
    img *= 255;
    cv::rectangle(img, bbForeground, cv::Scalar(0, 0, 0));
    cv::rectangle(img, bbBackground, cv::Scalar(0, 0, 0));
    img.convertTo(img, CV_8UC3);
    cv::imshow("Color Image", img);

    cv::waitKey();

    return 0;
}