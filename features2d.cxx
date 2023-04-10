#include "features2d.h"

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


#include "point3d.h"


 std::string ImageDescriptor::serialize() const
 {
        std::stringstream ss;
        ss << keypoints.size() << "\n";
        for (auto& kp : keypoints) {
            ss << kp.pt.x << " " << kp.pt.y << " " << kp.size << "\n";
        }
        ss << descriptors.cols << "\n";
        for (int i = 0; i < descriptors.rows; ++i) {
            for (int j = 0; j < descriptors.cols; ++j) {
                ss << static_cast<int>(descriptors.at<uint8_t>(i, j)) << " ";
            }
            ss << "\n";
        }
        return ss.str();
}

ImageDescriptor::Ptr ImageDescriptor::deserialize(const std::string& serial)
{
    auto obj = std::make_shared<ImageDescriptor>();
    std::istringstream iss(serial);
    int n;
    iss >> n;
    for (int i = 0; i < n; ++i) {
        float x, y, s;
        iss >> x >> y >> s;
        obj->keypoints.push_back(cv::KeyPoint(x, y, s));
    }
    int m;
    iss >> m;
    obj->descriptors.create(n, m, CV_8U);
    int c;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            iss >> c;
            obj->descriptors.at<uint8_t>(i, j) = static_cast<uint8_t>(c);
        }
    }

    return obj;
}

ImageDescriptor::Ptr computeImageDescriptors(cv::Mat img, int nKps) {
    ImageDescriptor::Ptr desc = ImageDescriptor::create();
    double scaleFactor = 1.5;
    // cv::Ptr<cv::ORB> detector = cv::ORB::create(nKps, scaleFactor);
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
    detector->detectAndCompute(img, cv::Mat(), desc->keypoints, desc->descriptors);

    return desc;
}

void ImageDescriptor::computeNormalizedKeypoints(const Eigen::Matrix3d& K) {
    keypoints_norm.clear();
    keypoints_norm.resize(keypoints.size());
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);
    for (int i = 0; i < keypoints.size(); ++i) {
        keypoints_norm[i][0] = (keypoints[i].pt.x - cx) / fx;
        keypoints_norm[i][1] = (keypoints[i].pt.y - cy) / fy;
        keypoints_norm[i][2] = 1.0;
    }
}

void Feature2d::setPointRef(Point3d *p3) {
    point_ref_ = p3;
    p3->addObservation(this);
}