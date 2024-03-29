#include "frame.h"

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "io.h"


void Frame::detectAndDescribe(int nKps)
{
    double scaleFactor = 1.5;
    // cv::Ptr<cv::ORB> detector = cv::ORB::create(nKps, scaleFactor);
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    detector->detectAndCompute(img_, cv::Mat(), kps, desc);

    for (int i = 0; i < std::min((int)kps.size(), nKps); ++i) {
        Eigen::Vector2d p(kps[i].pt.x, kps[i].pt.y);
        features_.push_back(Feature2d::create(p, this));
        keypoints_.push_back(kps[i]);
    }
    descriptors_ = desc.rowRange(0, std::min((int)kps.size(), nKps));
}







std::string Frame::serializeToOBJ(double size, int n_points)
{
    Matrix3d orientation = Rt_.block<3, 3>(0, 0).transpose();
    Vector3d position = -orientation * Rt_.col(3);
    return serializeTriaxeOBJ(orientation, position, size, n_points);
}