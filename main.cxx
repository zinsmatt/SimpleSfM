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
#include "optimize.h"
#include "triangulation.h"
#include "utils.h"

using namespace std;


MatchList::Ptr invertMatchList(MatchList::Ptr matches) {
    MatchList::Ptr new_list = MatchList::create();
    for (auto &m : *matches) {
        new_list->push_back(cv::DMatch(m.trainIdx, m.queryIdx, m.distance));
    }
    return new_list;
}



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
    for (int i = 0; i < 11; ++i) {
    // for (int i = 20; i < 150; i+=20) {
        char buffer[1024];
        // std::sprintf(buffer, "../data/herzjesu/%04d.png", i);
        // std::sprintf(buffer, "../data/MI9T/video_01/frame_%06d.png", i);
        std::sprintf(buffer, "../data/fountain/%04d.png", i);
        std::string name(buffer);
        // std::string name = filenames[i];
        cv::Mat img_init = cv::imread(name, cv::IMREAD_COLOR);
        cv::Mat img;
        cv::resize(img_init, img, cv::Size(img_init.size().width * SCALING, img_init.size().height * SCALING));

        auto frame = Frame::create(img, intrinsics);
        frame->detectAndDescribe(2000);
    
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
            RobustMatcher matcher(0.3);
            auto matches = matcher.robustMatch(frames[i], frames[j]);
            match_matrix[i][j] = matches;
            match_matrix[j][i] = invertMatchList(matches);

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
    Matrix34d Rt0;
    Rt0 << Matrix3d::Identity(), Vector3d::Zero();
    frames[init_idx_0]->setRt(Rt0);
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

    std::cout << "Initial cameras: " << init_idx_0 << " " << init_idx_1 << "\n";
    Map map;
    triangulateFeatures(frames[init_idx_0], frames[init_idx_1],
                        match_matrix[init_idx_0][init_idx_1], map, 1000);

    std::set<int> localized_frame_indices;
    localized_frame_indices.insert(init_idx_0);
    localized_frame_indices.insert(init_idx_1);


    std::vector<Frame::Ptr> new_cameras;
    
    for (int k = 0; k < frames.size()-2; ++k) {
        int max_nb_matches = 0;
        int best_new_camera_idx = -1;
        std::vector<cv::Point2d> best_points2d;
        std::vector<cv::Point3d> best_points3d;
        for (int i = 0; i < frames.size(); ++i) {
            if (!frames[i]->isLocalized()) {
                std::vector<cv::Point2d> points2d;
                std::vector<cv::Point3d> points3d;
                for (auto j : localized_frame_indices) {
                    MatchList::Ptr matches = match_matrix[i][j];
                    for (auto &m : *matches) {
                        auto fi = frames[i]->getFeature(m.queryIdx);
                        auto fj = frames[j]->getFeature(m.trainIdx);
                        auto* point3d_ref = fj->getPointRef();
                        if (point3d_ref) {
                            points3d.push_back(eigen2cvpoint(point3d_ref->getPos()));
                            points2d.push_back(eigen2cvpoint(fi->getPos()));
                        }
                    }
                }

                if (points2d.size() > max_nb_matches) {
                    max_nb_matches = points2d.size();
                    best_new_camera_idx = i;
                    best_points2d = std::move(points2d);
                    best_points3d = std::move(points3d);
                }
            }
        }

        if (best_new_camera_idx >= 0 && max_nb_matches >= 4) {
            // cv::solvePnP()
            std::cout << "Best new camera " << best_new_camera_idx << "\n";
            std::cout << "Matches " << best_points2d.size() << " 2D points and " << best_points3d.size() << " 3D points\n";
            cv::Mat t, rvec, R;
            cv::solvePnP(best_points3d, best_points2d, eigen2cv(K), cv::Mat(), rvec, t, false, cv::SOLVEPNP_EPNP);
            cv::Rodrigues(rvec, R);
            std::cout << "R \n" << R << "\n" << "t\n" << t << "\n\n";

            Matrix34d Rt;
            Rt << cv2eigen<double, 3, 3>(R), cv2eigen<double, 3, 1>(t);
            frames[best_new_camera_idx]->setRt(Rt);

            new_cameras.push_back(frames[best_new_camera_idx]);

            for (auto idx : localized_frame_indices) {
                std::cout << ">>> Triangulate new points between " << best_new_camera_idx << " and " << idx << "\n";
                triangulateFeatures(frames[best_new_camera_idx], frames[idx], match_matrix[best_new_camera_idx][idx], map, 1000);
            }
            localized_frame_indices.insert(best_new_camera_idx);
        }
        std::vector<Frame::Ptr> subset_frames;
        for (auto idx : localized_frame_indices)
            subset_frames.push_back(frames[idx]);
        runBundleAdjustment(subset_frames, map, init_idx_0, 10);   
    }


    // map.analyze();

    map.saveOBJ("reconstructed_map.obj");

    writeOBJ("reconstruction.obj", triangulated_points);
    writeTriaxeOBJ("cameras.obj", {Eigen::Matrix3d::Identity(), Rt.block<3, 3>(0, 0).transpose()}, {Eigen::Vector3d::Zero(), -Rt.block<3, 3>(0, 0).transpose() * Rt.col(3)}, 1.5, 100);

    saveCamerasOBJ("cameras.obj", frames, 1.0, 100);

    
    runBundleAdjustment(frames, map, init_idx_0, 50);   

    saveCamerasOBJ("optim_cameras.obj", frames, 1.0, 100);
    map.saveOBJ("optim_reconstructed_map.obj");


    for (int k = 0; k < frames.size(); ++k) {
        auto f = frames[k];
        for (int i = 0; i < map.size(); ++i) {
            auto X = map.getPoint(i)->getPos();
            Vector3d uvw = f->getP() * X.homogeneous();
            uvw /= uvw.z();
            Vector2d uv = uvw.head<2>();
            cv::circle(f->getImg(), eigen2cvpoint(uv), 1.0, cv::Scalar(0.0, 255, 0.0), cv::FILLED);
        }

        cv::imwrite("frame_" + std::to_string(k) + ".png", f->getImg());
    }

    return 0;
}