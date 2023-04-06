#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <memory>
#include <Eigen/Dense>

#include "features2d.h"
#include "io.h"
#include "matching.h"

using namespace std;




int main() {

    std::vector<cv::Mat> images;
    std::vector<ImageDescriptor::Ptr> imageDescriptors;

    for (int i = 0; i <= 7; ++i) {
        char buffer[1024];
        std::sprintf(buffer, "../data/herzjesu/%04d.png", i);
        std::string name(buffer);
        cv::Mat img_init = cv::imread(name);
        cv::Mat img;
        cv::resize(img_init, img, cv::Size(img_init.size().width * 0.3, img_init.size().height * 0.3));
        images.push_back(img);


        // auto desc = computeImageDescriptors(img);
        // std::cout << desc->descriptors.type() << "\n";
        // imageDescriptors.push_back(desc);
        // std::cout << desc->keypoints.size() << " poins detected\n";

        // cv::Mat img_keypoints;
        // cv::drawKeypoints(img, desc->keypoints, img_keypoints);
        // cv::imshow("keypoints", img_keypoints);
        // cv::waitKey();
    }

    // saveImageDescriptors("image_descriptors.txt", imageDescriptors);

    loadImageDescriptors("image_descriptors.txt", imageDescriptors);

    for (int i = 0; i < imageDescriptors.size(); ++i) {
        auto d = imageDescriptors[i];
        auto d2 = imageDescriptors[i];
        std::cout << d->keypoints.size() << "\n";
        std::cout << d2->keypoints.size() << "\n";

        for (int j = 0; j < d->keypoints.size(); ++j) {
            auto kp = d->keypoints[j];
            auto kp2 = d2->keypoints[j];
            if (std::abs(kp.pt.x - kp2.pt.x) > 1.0e-3 ||
                std::abs(kp.pt.y - kp2.pt.y) > 1.0e-3 ||
                std::abs(kp.size - kp2.size) > 1.0e-3)
                std::cout << "ERROR keypoints\n";
        }


        std::cout << d->descriptors.size << "\n";
        std::cout << d2->descriptors.size << "\n";
        for (int ii = 0; ii < d->descriptors.rows; ++ii) {
            for (int j = 0; j < d->descriptors.cols; ++j) {
                auto a = d->descriptors.at<uint8_t>(ii, j);
                auto a2 = d2->descriptors.at<uint8_t>(ii, j);
                if (a != a2)
                    std::cout << ii << " " << j <<  " : ERROR descriptor\n";
            }
        }

    }

    for (int i = 0; i <= 7; ++i)
    {
        int id0 = i;
        int id1 = i+1;
        RobustMatcher matcher(0.5);
        auto matches = matcher.robustMatch(imageDescriptors[id0], imageDescriptors[id1]);

        cv::Mat img_matches;
        cv::drawMatches(images[id0], imageDescriptors[id0]->keypoints, images[id1], imageDescriptors[id1]->keypoints,
                        matches, img_matches);
        
        cv::imshow("matches", img_matches);
        cv::waitKey();
    }


    Eigen::Matrix3d K;
    K << 2759.48, 0.0, 1520.69,
         0.0, 2764.16, 1006.81,
         0.0, 0.0, 1.0;









    return 0;
}