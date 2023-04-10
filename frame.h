#pragma once

#include <opencv2/opencv.hpp>

#include "camera.h"
#include "features2d.h"


class Frame
{
    cv::Mat img_;
    Matrix34d Rt_;
    Intrinsics::Ptr calib_;
    std::vector<Feature2d::Ptr> features_;
    cv::Mat descriptors_;
    std::vector<cv::KeyPoint> keypoints_;

    bool localized_ = false;


public:
    typedef std::shared_ptr<Frame> Ptr;

    static Ptr create(cv::Mat img, Intrinsics::Ptr calib) {
        return std::make_shared<Frame>(img, calib);
    }

    Frame(cv::Mat img, Intrinsics::Ptr calib) : img_(img), calib_(calib) {
        Rt_.block<3, 3>(0, 0) = Matrix3d::Identity();
    }

    bool isLocalized() const {
        return localized_;
    }

    void setRt(const Matrix34d& Rt) {
        Rt_ = Rt;
        localized_ = true;
    }

    Matrix34d getRt() const {
        if (!localized_) {
            std::cerr << "Warning !! the frame has not been localized yet." << std::endl;
        }
        return Rt_;
    }

    Matrix3d getK() const {
        return calib_->K;
    }

    Matrix34d getP() const {
        return calib_->K * Rt_;
    }

    void detectAndDescribe(int nKps=2000);


    cv::Mat getDescriptors() {
        return descriptors_;
    }

    std::vector<cv::KeyPoint> getcvKeypoints() const {
        std::vector<cv::KeyPoint> kps(features_.size());
        for (int i = 0; i < kps.size(); ++i) {
            kps[i] = cv::KeyPoint(eigen2cvpoint(features_[i]->getPos()), 1.0);
        }
        return kps;
    }

    std::vector<cv::Point2d> getcvPoints() const {
        std::vector<cv::Point2d> pts(features_.size());
        for (int i = 0; i < pts.size(); ++i) {
            pts[i] = eigen2cvpoint(features_[i]->getPos());
        }
        return pts;
    }

    std::vector<Vector2d> getFeaturePoints() const {
        std::vector<Vector2d> pts(features_.size());
        for (int i = 0; i < pts.size(); ++i) {
            pts[i] = features_[i]->getPos();
        }
        return pts;
    }

    std::vector<Feature2d::Ptr>& getFeatures() {
        return features_;
    }

    Feature2d::Ptr getFeature(int idx) {
        return features_[idx];
    }

    cv::Mat getImg() {
        return img_;
    }

    std::string serializeToOBJ(double size, int n_points);

};