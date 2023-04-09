#pragma once

#include <opencv2/opencv.hpp>

#include "camera.h"
#include "features2d.h"


class Frame
{
    cv::Mat img_;
    Matrix34d Rt_;
    std::shared_ptr<Intrinsics> calib_;
    std::vector<Feature2d::Ptr> features_;
    cv::Mat descriptors_;
    std::vector<cv::KeyPoint> keypoints_;

    bool localized_ = false;


public:
    typedef std::shared_ptr<Frame> Ptr;

    static Ptr create(cv::Mat img) {
        return std::make_shared<Frame>(img);
    }

    Frame(cv::Mat img) : img_(img) {}

    bool isLocalized() const {
        return localized_;
    }

    void setRt(const Matrix34d& Rt) {
        Rt_ = Rt;
        localized_ = true;
    }

    Matrix34d getRt() const {
        return Rt_;
    }

    void detectAndDescribe(int nKps=2000);


    cv::Mat getDescriptors() {
        return descriptors_;
    }

    std::vector<cv::KeyPoint> getcvKeypoints() const {
        std::vector<cv::KeyPoint> kps(features_.size());
        for (int i = 0; i < kps.size(); ++i) {
            kps[i] = cv::KeyPoint(eigen2cvpoint(features_[i]->pos()), 1.0);
        }
        return kps;
    }

    std::vector<cv::Point2d> getcvPoints() const {
        std::vector<cv::Point2d> pts(features_.size());
        for (int i = 0; i < pts.size(); ++i) {
            pts[i] = eigen2cvpoint(features_[i]->pos());
        }
        return pts;
    }

    std::vector<Vector2d> getFeaturePoints() const {
        std::vector<Vector2d> pts(features_.size());
        for (int i = 0; i < pts.size(); ++i) {
            pts[i] = features_[i]->pos();
        }
        return pts;
    }

    cv::Mat getImg() {
        return img_;
    }

};