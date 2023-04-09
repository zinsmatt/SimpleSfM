#pragma once

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

#include <memory>

class Frame;

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


class Feature2d
{
    Eigen::Vector2d pos_;
    Frame* frame_ = nullptr;


public:
    typedef std::shared_ptr<Feature2d> Ptr;

    static Ptr create(const Eigen::Vector2d& p, Frame* frame) {
        return std::make_shared<Feature2d>(p, frame);
    }

    Feature2d(const Eigen::Vector2d& p, Frame* frame) :
        pos_(p), frame_(frame) {}

    Eigen::Vector2d pos() const {
        return pos_;
    }

};