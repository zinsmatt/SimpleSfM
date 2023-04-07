#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <Eigen/Dense>

#include <memory>


struct ImageDescriptor
{
    typedef std::shared_ptr<ImageDescriptor> Ptr;
    static Ptr create() {
        return std::make_shared<ImageDescriptor>();
    }
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<Eigen::Vector3d> keypoints_norm;

    void computeNormalizedKeypoints(const Eigen::Matrix3d& K);

    std::string serialize() const;

    static Ptr deserialize(const std::string& serial);
};



ImageDescriptor::Ptr computeImageDescriptors(cv::Mat img, int nKps=2000);
