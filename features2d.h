#pragma once

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

#include <memory>

class Frame;
class Point3d;

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
    Frame* frame_ref_ = nullptr;
    Point3d* point_ref_ = nullptr;

public:
    typedef std::shared_ptr<Feature2d> Ptr;

    static Ptr create(const Eigen::Vector2d& p, Frame* frame) {
        return std::make_shared<Feature2d>(p, frame);
    }

    Feature2d(const Eigen::Vector2d& p, Frame* frame) :
        pos_(p), frame_ref_(frame) {}

    const Eigen::Vector2d& getPos() const {
        return pos_;
    }

    void setPointRef(Point3d *p3) {
        point_ref_ = p3;
    }

    Point3d* getPointRef() const {
        return point_ref_;
    }

};