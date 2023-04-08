#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <memory>
#include <Eigen/Dense>

#include "epipolar.h"
#include "features2d.h"
#include "io.h"
#include "matching.h"
#include "triangulation.h"
#include "utils.h"

using namespace std;




int main() {
    // double SCALING = 0.3;
    // double SCALING = 0.3;
    double SCALING = 0.5;
    std::vector<cv::Mat> images;
    std::vector<ImageDescriptor::Ptr> imageDescriptors;

    // std::vector<std::string> filenames = {"../data/fountain/0004.png", "../data/fountain/0005.png"};
    std::vector<std::string> filenames = {"../data/herzjesu/0001.png", "../data/herzjesu/0002.png"};
    // std::vector<std::string> filenames = {"../data/MI9T/video_04/frame_000150.png", "../data/MI9T/video_04/frame_000170.png"};
    // std::vector<std::string> filenames = {"../data/room/images/frame_000490.png", "../data/room/images/frame_000510.png"};

    for (int i = 0; i < filenames.size(); ++i) {
        // char buffer[1024];
        // std::sprintf(buffer, "../data/herzjesu/%04d.png", i);
        // std::string name(buffer);
        std::string name = filenames[i];
        cv::Mat img_init = cv::imread(name, cv::IMREAD_COLOR);
        cv::Mat img;
        cv::resize(img_init, img, cv::Size(img_init.size().width * SCALING, img_init.size().height * SCALING));
        images.push_back(img);


        auto desc = computeImageDescriptors(img, 2000);
        std::cout << desc->descriptors.type() << "\n";
        imageDescriptors.push_back(desc);
        std::cout << desc->keypoints.size() << " poins detected\n";

        // cv::Mat img_keypoints;
        // cv::drawKeypoints(img, desc->keypoints, img_keypoints);
        // cv::imshow("keypoints", img_keypoints);
        // cv::waitKey();
    }

    // saveImageDescriptors("image_descriptors.txt", imageDescriptors);

    // loadImageDescriptors("image_descriptors.txt", imageDescriptors);

    
    std::vector<std::vector<cv::DMatch>> list_matches;
    for (int i = 0; i < imageDescriptors.size()-1; ++i)
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

        list_matches.push_back(std::move(matches));
    }


    Eigen::Matrix3d K;
    K << 2759.48 * SCALING, 0.0, 1520.69 * SCALING,
         0.0, 2764.16 * SCALING, 1006.81 * SCALING,
         0.0, 0.0, 1.0;

    // K << 565.077 * SCALING, 0.0, 320 * SCALING,
    //      0.0, 565.077 * SCALING, 180 * SCALING,
    //      0.0, 0.0, 1.0;


    // K << 565.077, 0.0, 320,
    //      0.0, 565.077, 180,
    //      0.0, 0.0, 1.0;

    for (auto desc : imageDescriptors) {
        desc->computeNormalizedKeypoints(K);
    }

    int idx = 0;

    Eigen::Matrix3d E = computeEssentialMatrix(imageDescriptors[idx], imageDescriptors[idx+1], list_matches[idx], K);
    std::cout << "Essential matrix:\n" << E << "\n";

    std::vector<Eigen::Vector3d> triangulated_points;
    auto Rt = computeRtFromEssential(E, imageDescriptors[idx], imageDescriptors[idx+1], list_matches[idx], K, triangulated_points, 1000.0);



    std::cout << "comouteRtFromEssential: \n" << Rt << "\n";


    std::vector<cv::KeyPoint> proj_pts_1;
    std::vector<cv::KeyPoint> proj_pts_2;
    for (auto& X : triangulated_points) {

        Eigen::Vector3d p1 = K * X;
        p1 /= p1.z();


        Eigen::Vector3d p2 = K * (Rt * X.homogeneous());
        p2 /= p2.z();


        cv::KeyPoint kp1(cv::Point2d(p1.x(), p1.y()), 1.0);
        proj_pts_1.push_back(kp1);
        cv::KeyPoint kp2(cv::Point2d(p2.x(), p2.y()), 1.0);
        proj_pts_2.push_back(kp2);

    }


    cv::Mat img_1_reproj;
    std::vector<cv::KeyPoint> subset_kps1;
    std::vector<cv::KeyPoint> subset_kps2;
    for (auto &m : list_matches[idx]) {
        subset_kps1.push_back(imageDescriptors[idx]->keypoints[m.queryIdx]);
        subset_kps2.push_back(imageDescriptors[idx+1]->keypoints[m.trainIdx]);
    }
    cv::drawKeypoints(images[idx], subset_kps1, img_1_reproj, cv::Scalar(0, 0, 255));
    cv::drawKeypoints(img_1_reproj, proj_pts_1, img_1_reproj, cv::Scalar(0, 255, 0));
    cv::imshow("reproj 1", img_1_reproj);
    cv::waitKey();


    cv::Mat img_2_reproj;
    cv::drawKeypoints(images[idx+1], subset_kps2, img_2_reproj, cv::Scalar(0, 0, 255));
    cv::drawKeypoints(img_2_reproj, proj_pts_2, img_2_reproj, cv::Scalar(0, 255, 0));
    cv::imshow("reproj 2", img_2_reproj);
    cv::waitKey();


    writeOBJ("reconstruction.obj", triangulated_points);
    writeTriaxeOBJ("cameras.obj", {Eigen::Matrix3d::Identity(), Rt.block<3, 3>(0, 0).transpose()}, {Eigen::Vector3d::Zero(), -Rt.block<3, 3>(0, 0).transpose() * Rt.col(3)}, 1.5, 100);

    return 0;
}