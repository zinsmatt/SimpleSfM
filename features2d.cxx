#include "features2d.h"



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

ImageDescriptor::Ptr computeImageDescriptors(cv::Mat img) {
    ImageDescriptor::Ptr desc = ImageDescriptor::create();
    double nKeypoints = 2000;
    double scaleFactor = 1.5;
    cv::Ptr<cv::ORB> detector = cv::ORB::create(nKeypoints, scaleFactor);
    detector->detectAndCompute(img, cv::Mat(), desc->keypoints, desc->descriptors);
    return desc;
}