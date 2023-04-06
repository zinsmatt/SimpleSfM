#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <memory>


struct ImageDescriptor
{
    typedef std::shared_ptr<ImageDescriptor> Ptr;
    static Ptr create() {
        return std::make_shared<ImageDescriptor>();
    }
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;

    std::string serialize() const;

    static Ptr deserialize(const std::string& serial);
};



ImageDescriptor::Ptr computeImageDescriptors(cv::Mat img);
