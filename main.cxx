#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <memory>
#include <Eigen/Dense>

#include "epipolar.h"
#include "features2d.h"
#include "frame.h"
#include "io.h"
#include "map.h"
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
    std::vector<Frame::Ptr> frames;

    std::vector<std::string> filenames = {"../data/fountain/0001.png", "../data/fountain/0002.png"};
    // std::vector<std::string> filenames = {"../data/herzjesu/0001.png", "../data/herzjesu/0002.png"};
    // std::vector<std::string> filenames = {"../data/MI9T/video_04/frame_000150.png", "../data/MI9T/video_04/frame_000170.png"};
    // std::vector<std::string> filenames = {"../data/room/images/frame_000490.png", "../data/room/images/frame_000510.png"};


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

    // for (auto desc : imageDescriptors) {
    //     desc->computeNormalizedKeypoints(K);
    // }


    Intrinsics::Ptr intrinsics = Intrinsics::create(K);

    // for (int i = 0; i < filenames.size(); ++i) {
    for (int i = 0; i < 5; ++i) {
        char buffer[1024];
        // std::sprintf(buffer, "../data/herzjesu/%04d.png", i);
        std::sprintf(buffer, "../data/fountain/%04d.png", i);
        std::string name(buffer);
        // std::string name = filenames[i];
        cv::Mat img_init = cv::imread(name, cv::IMREAD_COLOR);
        cv::Mat img;
        cv::resize(img_init, img, cv::Size(img_init.size().width * SCALING, img_init.size().height * SCALING));

        auto frame = Frame::create(img, intrinsics);
        frame->detectAndDescribe(50000);
    
        frames.push_back(frame);
        // images.push_back(img);


        // auto desc = computeImageDescriptors(img, 2000);
        // std::cout << desc->descriptors.type() << "\n";
        // imageDescriptors.push_back(desc);
        // std::cout << desc->keypoints.size() << " poins detected\n";

        // cv::Mat img_keypoints;
        // cv::drawKeypoints(frame->getImg(), frame->getcvKeypoints(), img_keypoints);
        // cv::imshow("keypoints", img_keypoints);
        // cv::waitKey();
    }



    // saveImageDescriptors("image_descriptors.txt", imageDescriptors);

    // loadImageDescriptors("image_descriptors.txt", imageDescriptors);

    
    int max_matches = 0;
    int max_matches_i = -1;
    int max_matches_j = -1;
    std::vector<std::vector<MatchList::Ptr>> match_matrix(frames.size(), std::vector<MatchList::Ptr>(frames.size()));
    for (int i = 0; i < frames.size()-1; ++i)
    {
        for (int j = i+1; j < frames.size(); ++j) {
            std::cout << "Match images " << i << " " <<  j << ":\n";
            RobustMatcher matcher(0.5);
            auto matches = matcher.robustMatch(frames[i], frames[j]);
            match_matrix[i][j] = matches;

            if (matches->size() > max_matches) {
                max_matches = matches->size();
                max_matches_i = i;
                max_matches_j = j;
            }

            // if (matches)

            // cv::Mat img_matches;
            // cv::drawMatches(frames[i]->getImg(), frames[i]->getcvKeypoints(),
            //                 frames[j]->getImg(), frames[j]->getcvKeypoints(),
            //                 matches->matches, img_matches);
            
            // cv::imshow("matches", img_matches);
            // cv::waitKey();
        }
    }



    int init_idx_0 = max_matches_i;
    int init_idx_1 = max_matches_j;


    Eigen::Matrix3d E = computeEssentialMatrix(frames[init_idx_0]->getcvPoints(),
                                               frames[init_idx_1]->getcvPoints(),
                                               *match_matrix[init_idx_0][init_idx_1], K);
    std::cout << "Essential matrix:\n" << E << "\n";



    std::vector<Eigen::Vector3d> triangulated_points;
    auto Rt = computeRtFromEssential(E, frames[init_idx_0]->getFeaturePoints(),
                                        frames[init_idx_1]->getFeaturePoints(),
                                        *match_matrix[init_idx_0][init_idx_1],
                                        K, triangulated_points, 1000.0);
    frames[init_idx_1]->setRt(Rt);



    std::cout << "comouteRtFromEssential: \n" << Rt << "\n";



    // Check projected triangulated points

    // std::vector<cv::KeyPoint> proj_pts_1;
    // std::vector<cv::KeyPoint> proj_pts_2;
    // std::cout << "tri points " << triangulated_points.size() << std::endl;
    // for (auto& X : triangulated_points) {

    //     Eigen::Vector3d p1 = K * X;
    //     p1 /= p1.z();


    //     Eigen::Vector3d p2 = K * (Rt * X.homogeneous());
    //     p2 /= p2.z();

 
    //     cv::KeyPoint kp1(cv::Point2d(p1.x(), p1.y()), 1.0);
    //     proj_pts_1.push_back(kp1);
    //     cv::KeyPoint kp2(cv::Point2d(p2.x(), p2.y()), 1.0);
    //     proj_pts_2.push_back(kp2);

    // }


    // cv::Mat img_1_reproj;
    // auto kps1 = frames[idx]->getcvKeypoints();
    // auto kps2 = frames[idx+1]->getcvKeypoints();
    // std::vector<cv::KeyPoint> subset_kps1;
    // std::vector<cv::KeyPoint> subset_kps2;
    // for (auto &m : match_matrix[idx][idx+1]->matches) {
    //     subset_kps1.push_back(kps1[m.queryIdx]);
    //     subset_kps2.push_back(kps2[m.trainIdx]);
    // }
    // cv::drawKeypoints(frames[idx]->getImg(), subset_kps1, img_1_reproj, cv::Scalar(0, 0, 255));
    // cv::drawKeypoints(img_1_reproj, proj_pts_1, img_1_reproj, cv::Scalar(0, 255, 0));
    // cv::imshow("reproj 1", img_1_reproj);
    // cv::waitKey();


    // cv::Mat img_2_reproj;
    // cv::drawKeypoints(frames[idx+1]->getImg(), subset_kps2, img_2_reproj, cv::Scalar(0, 0, 255));
    // cv::drawKeypoints(img_2_reproj, proj_pts_2, img_2_reproj, cv::Scalar(0, 255, 0));
    // cv::imshow("reproj 2", img_2_reproj);
    // cv::waitKey();

    Map map;
    triangulateFeatures(frames[init_idx_0], frames[init_idx_1],
                        match_matrix[init_idx_0][init_idx_1], map, 1000);

    map.saveMap("reconstructed_map.obj");

    for (int i = 0; i < frames.size(); ++i) {
        if (i != init_idx_0 && i != init_idx_1) {
            // cv::solvePnP()
        }
    }

    writeOBJ("reconstruction.obj", triangulated_points);
    writeTriaxeOBJ("cameras.obj", {Eigen::Matrix3d::Identity(), Rt.block<3, 3>(0, 0).transpose()}, {Eigen::Vector3d::Zero(), -Rt.block<3, 3>(0, 0).transpose() * Rt.col(3)}, 1.5, 100);

    return 0;
}