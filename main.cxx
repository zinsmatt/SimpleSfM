#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <memory>
#include <Eigen/Dense>

#include "features2d.h"
#include "io.h"
#include "matching.h"
#include "triangulation.h"

using namespace std;


void writeOBJ(const std::string& filename, const std::vector<Eigen::Vector3d>& points) {
    std::ofstream fout(filename);
    for (auto& p : points) {
        fout << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    fout.close();
}

Eigen::Matrix3d computeEssentialMatrix(ImageDescriptor::Ptr desc1, ImageDescriptor::Ptr desc2,
                                       const std::vector<cv::DMatch>& matches, const Eigen::Matrix3d& K) {

    std::vector<cv::Point2d> pts1, pts2;
    for (int i = 0; i < matches.size(); ++i) {
        auto uv1 = desc1->keypoints[matches[i].queryIdx];
        auto uv2 = desc2->keypoints[matches[i].trainIdx];
        pts1.push_back(uv1.pt);
        pts2.push_back(uv2.pt);
    }

    cv::Mat KK(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    KK.at<double>(i, j) = K(i, j);

    cv::Mat EE = cv::findEssentialMat(pts1, pts2, KK, cv::RANSAC, 0.9999, 1.0); //, cv::LMEDS);





    Eigen::Matrix3d EEE;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EEE(i, j) = EE.at<double>(i, j);

    EEE /= EEE(2, 2);
    return EEE;



    std::cout << "====================\n" << EEE << "\n=======================\n\n";










    std::cout << "Estimate Essential matrix from " << matches.size() << " correspondences.\n";

    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);
    // std::vector<Eigen::Vector2d> X1s(matches.size()), X2s(matches.size());
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(matches.size(), 9);
    for (int i = 0; i < matches.size(); ++i) {
        auto uv1 = desc1->keypoints_norm[matches[i].queryIdx];
        auto uv2 = desc2->keypoints_norm[matches[i].trainIdx];

        double u1 = uv1[0];
        double v1 = uv1[1];
        double u2 = uv2[0];
        double v2 = uv2[1];

        // X1s[i][0] = u1;
        // X1s[i][1] = v1;
        // X2s[i][0] = u2;
        // X2s[i][1] = v2;

        A(i, 0) = u1 * u2;
        A(i, 1) = v1 * u2;
        A(i, 2) =  u2;

        A(i, 3) = u1 * v2;
        A(i, 4) = v1 * v2;
        A(i, 5) =  v2;


        A(i, 6) = u1;
        A(i, 7) = v1;
        A(i, 8) = 1;
    }

    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(A,  Eigen::ComputeFullV);

    // cout << "Its singular values are:" << endl << svd.singularValues() << endl;
    // std::cout << svd.matrixV() << "\n";
    auto v = svd.matrixV().col(8);
    Eigen::Matrix3d E;
    E << v[0], v[1], v[2],
         v[3], v[4], v[5], 
         v[6], v[7], v[8];

    Eigen::JacobiSVD<Eigen::Matrix3d> svdE(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singvals = svdE.singularValues();
    // std::cout << "E singular values:\n" <<  singvals << "\n";


    Eigen::DiagonalMatrix<double, 3> S(singvals[0], singvals[1], 0.0);

    Eigen::Matrix3d E_better =  svdE.matrixU() * S * svdE.matrixV().transpose();
    E_better /= E_better(2, 2);
    // std::cout << "E:\n" << E << "\n";
    std::cout << "E_better:\n" << E_better << "\n";


    double err = 0, err_2 = 0;
    for (int i = 0; i < matches.size(); ++i) {
        auto uv1 = desc1->keypoints_norm[matches[i].queryIdx];
        auto uv2 = desc2->keypoints_norm[matches[i].trainIdx];
        double e = uv2.transpose() * E_better * uv1;
        err += std::abs(e);
        double ee = uv2.transpose() * EEE * uv1;
        err_2 += std::abs(ee);

        std::cout << "err = " << e
        << "   |   " <<  ee << "\n";
    }
    std::cout << "ERRORS " << err / matches.size() << " " << err_2/matches.size() << "\n";


    Eigen::JacobiSVD<Eigen::Matrix3d> svdE_better(E_better, Eigen::ComputeFullU | Eigen::ComputeFullV);
    std::cout << "sing values E_better: " << svdE_better.singularValues() << "\n";

    Eigen::JacobiSVD<Eigen::Matrix3d> svdEEE(EEE, Eigen::ComputeFullU | Eigen::ComputeFullV);
    std::cout << "sing values EEE: " << svdEEE.singularValues() << "\n";


    return EEE;
}

Eigen::Matrix<double, 3, 4>
 computeRtFromEssential(const Eigen::Matrix3d& E, ImageDescriptor::Ptr &desc1,
                            ImageDescriptor::Ptr desc2, std::vector<cv::DMatch>& matches,
                            const Eigen::Matrix3d& K, std::vector<Eigen::Vector3d>& triangulatedPoints)
{

    std::vector<cv::Point2d> pts1, pts2;
    for (int i = 0; i < matches.size(); ++i) {
        auto uv1 = desc1->keypoints[matches[i].queryIdx];
        auto uv2 = desc2->keypoints[matches[i].trainIdx];
        pts1.push_back(uv1.pt);
        pts2.push_back(uv2.pt);
    }


    cv::Mat EE(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    EE.at<double>(i, j) = E(i, j);
    cv::Mat KK(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    KK.at<double>(i, j) = K(i, j);

    std::cout << "\n" << KK << "\n";

    cv::Mat RR, tt;
    cv::Mat tri_points;
    cv::Mat mask(cv::Size2i(1, pts1.size()), CV_8U, 1);
    cv::recoverPose(EE, pts1, pts2, KK, RR, tt, 0.01, cv::noArray(), tri_points);
    std::cout << "Mask : \n";
    std::cout << mask << "\n\n";

    for (int i = 0; i < tri_points.cols; ++i) {
        if (mask.at<uchar>(i, 0) == 0)
            continue;
        Eigen::Vector3d v(tri_points.at<double>(0, i), 
                          tri_points.at<double>(1, i), 
                          tri_points.at<double>(2, i));
        v /= tri_points.at<double>(3, i);
        std::cout << v.transpose() << "\n";
        triangulatedPoints.push_back(v);
    }

    Eigen::Vector3d ttt(tt.at<double>(0, 0), tt.at<double>(1, 0), tt.at<double>(2, 0));
    ttt.normalize();
    Eigen::Matrix3d RRR;
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    RRR(i, j) = RR.at<double>(i, j);

    std::cout << "=========================\n";
    std::cout << "ttt = " << ttt.transpose() << "\n";
    std::cout << "RRR = " << RRR << "\n";
    std::cout << "=========================\n";
    Eigen::Matrix<double, 3, 4> res;
    res.block<3, 3>(0, 0) = RRR;
    res.col(3) = ttt;
    return res;


    





    Eigen::JacobiSVD<Eigen::Matrix3d> svdE(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svdE.matrixU();
    Eigen::Matrix3d Vt = svdE.matrixV().transpose();
    if (U.determinant() < 0) 
        U *= -1;
    if (Vt.determinant() < 0)
        Vt *= -1;

    auto t = U.col(2);
    t.normalize();

    std::cout << "t = " << t.transpose() << "\n";

    Eigen::Matrix3d W;
    W << 0.0, -1.0, 0.0, 
         1.0, 0.0, 0.0,
         0.0, 0.0, 1.0;

    Eigen::Matrix3d R1 = U * W * Vt;
    Eigen::Vector3d t1 = t;

    Eigen::Matrix3d R2 = U * W * Vt;
    Eigen::Vector3d t2 = -t;


    Eigen::Matrix3d R3 = U * W.transpose() * Vt;
    Eigen::Vector3d t3 = t;

    Eigen::Matrix3d R4 = U * W.transpose() * Vt;
    Eigen::Vector3d t4 = -t;

    std::vector<Eigen::Matrix3d> possible_Rs = {R1, R2, R3, R4};
    std::vector<Eigen::Vector3d> possible_ts = {t1, t2, t3, t4};

    std::cout << "R1\n" << R1 << "\n";
    std::cout << "R2\n" << R2 << "\n";
    std::cout << "R3\n" << R3 << "\n";
    std::cout << "R4\n" << R4 << "\n";

    std::cout << "det R1 = " << R1.determinant() << "\n";


    int max_count_front = -1;
    int best_solution_idx = -1;

    for (int i = 0; i < 4; ++i) {
        Eigen::Matrix<double, 3, 4> Rt1;
        Rt1.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        
        Eigen::Matrix<double, 3, 4> Rt2;
        Rt2.block<3, 3>(0, 0) = possible_Rs[i];
        Rt2.col(3) = possible_ts[i];
        Rt2.block<3, 3>(0, 0) = RRR;
        Rt2.col(3) = ttt;

        Eigen::Matrix<double, 3, 4> P1 = K * Rt1;
        Eigen::Matrix<double, 3, 4> P2 = K * Rt2;

        int count_front = 0;
        std::vector<Eigen::Vector3d> tri_points;
        for (auto &m : matches) {
            auto tmp1 = desc1->keypoints[m.queryIdx].pt;
            Eigen::Vector2d p1(tmp1.x, tmp1.y);
            auto tmp2 = desc1->keypoints[m.trainIdx].pt;
            Eigen::Vector2d p2(tmp2.x, tmp2.y);

            Eigen::Vector3d X = triangulatePoint(p1, P1, p2, P2);

            if ((Rt1 * X.homogeneous()).z() > 0 &&
                (Rt2 * X.homogeneous()).z() > 0)
            {
                count_front++;
                tri_points.push_back(X);
            }
        }

        if (count_front > max_count_front) {
            max_count_front = count_front;
            best_solution_idx = i;
            triangulatedPoints = tri_points;
        }
    }

    std::cout << "Best solution has triangulated " << max_count_front << "/" << matches.size() << " points in front of the camera.\n";
    // Eigen::Isometry3d Rt;
    // Rt.translate(possible_ts[best_solution_idx]);
    // Rt.rotate(possible_Rs[best_solution_idx]);
    Eigen::Matrix<double, 3, 4> Rt;
    Rt.block<3 ,3>(0, 0) = possible_Rs[best_solution_idx];
    Rt.col(3) = possible_ts[best_solution_idx];
    // Eigen::Isometry3d Rt = Eigen::Translation3d(possible_ts[best_solution_idx]) * possible_Rs[best_solution_idx];

    // std::cout << Rt.translation() << "\n";
    // std::cout << Rt.rotation() << "\n";
    std::cout << "best indx = " << best_solution_idx << "\n";
    std::cout << "t = " <<  possible_ts[best_solution_idx].transpose() << "\n";
    std::cout <<   possible_Rs[best_solution_idx] << "\n";


    Rt.block<3 ,3>(0, 0) = RRR;
    Rt.col(3) = ttt;

    return Rt;    
}


int main() {
    // double SCALING = 0.3;
    double SCALING = 1;
    std::vector<cv::Mat> images;
    std::vector<ImageDescriptor::Ptr> imageDescriptors;

    std::vector<std::string> filenames = {"../data/MI9T/video_05/frame_000175.png", "../data/MI9T/video_05/frame_000200.png"};

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
    // K << 2759.48 * SCALING, 0.0, 1520.69 * SCALING,
    //      0.0, 2764.16 * SCALING, 1006.81 * SCALING,
    //      0.0, 0.0, 1.0;

    // K << 565.077 * SCALING, 0.0, 320 * SCALING,
    //      0.0, 565.077 * SCALING, 180 * SCALING,
    //      0.0, 0.0, 1.0;


    K << 765.077 * SCALING, 0.0, 320 * SCALING,
         0.0, 765.077 * SCALING, 180 * SCALING,
         0.0, 0.0, 1.0;

    for (auto desc : imageDescriptors) {
        desc->computeNormalizedKeypoints(K);
    }

    int idx = 0;

    Eigen::Matrix3d E = computeEssentialMatrix(imageDescriptors[idx], imageDescriptors[idx+1], list_matches[idx], K);

    std::vector<Eigen::Vector3d> triangulated_points;
    auto Rt = computeRtFromEssential(E, imageDescriptors[idx], imageDescriptors[idx+1], list_matches[idx], K, triangulated_points);




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

    std::cout << std::endl;

    writeOBJ("reconstruction.obj", triangulated_points);
    return 0;
}