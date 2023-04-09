#include "epipolar.h"

#include "triangulation.h"
#include "utils.h"
#include "io.h"


Eigen::Matrix3d computeEssentialMatrix(const std::vector<cv::Point2d>& pts0, const std::vector<cv::Point2d>& pts1,
                                       std::vector<cv::DMatch>& matches, const Eigen::Matrix3d& K)
{
    std::vector<cv::Point2d> matched_pts0(matches.size());
    std::vector<cv::Point2d> matched_pts1(matches.size());
    for (int i = 0; i < matches.size(); ++i) {
        matched_pts0[i] = pts0[matches[i].queryIdx];
        matched_pts1[i] = pts1[matches[i].trainIdx];
    }
    // Do matching
    cv::Mat mask;
    cv::Mat Ecv = cv::findEssentialMat(matched_pts0, matched_pts1, eigen2cv<double, 3, 3>(K), cv::RANSAC, 0.9999, 1.0, 1000, mask);

    // Filter bad matches
    std::vector<cv::DMatch> filtered_matches;
    for (int i = 0; i < matches.size(); ++i) {
        if (mask.at<uchar>(i, 0))
            filtered_matches.push_back(matches[i]);
    }
    matches = filtered_matches;

    return cv2eigen<double, 3, 3>(Ecv);
}
    

Matrix34d computeRtFromEssential(const Eigen::Matrix3d& E, const std::vector<Vector2d>& pts0,
                                 const std::vector<Vector2d>& pts1, std::vector<cv::DMatch>& matches,
                                 const Eigen::Matrix3d& K, std::vector<Eigen::Vector3d>& triangulated_points,
                                 double dmax)
{
    std::vector<Vector2d> matched_pts0(matches.size());
    std::vector<Vector2d> matched_pts1(matches.size());
    for (int i = 0; i < matches.size(); ++i) {
        matched_pts0[i] = pts0[matches[i].queryIdx];
        matched_pts1[i] = pts1[matches[i].trainIdx];
    }


    // Find the 4 solutions
    Eigen::JacobiSVD<Eigen::Matrix3d> svdE(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svdE.matrixU();
    Eigen::Matrix3d Vt = svdE.matrixV().transpose();
    if (U.determinant() < 0) 
        U *= -1;
    if (Vt.determinant() < 0)
        Vt *= -1;

    auto t = U.col(2);
    t.normalize();

    Eigen::Matrix3d W;
    W << 0.0, -1.0, 0.0, 
         1.0, 0.0, 0.0,
         0.0, 0.0, 1.0;

    Eigen::Matrix3d R1 = U * W * Vt;
    Eigen::Matrix3d R2 = U * W.transpose() * Vt;

    std::vector<Eigen::Matrix3d> possible_Rs = {R1, R1, R2, R2};
    std::vector<Eigen::Vector3d> possible_ts = {t, -t, t, -t};

    if (std::abs(R1.determinant()-1.0) > 1.0e-3 || std::abs(R2.determinant()-1.0) > 1.0e-3)
        std::cerr << "Error estimated rotation is bad. (determinant != 1)" << std::endl;


    // Chirality check to find the best solution
    int max_count_valid = -1;
    int best_solution_idx = -1;
    std::vector<int> best_solution_indices;

    Matrix34d P0;
    P0.block<3, 3>(0, 0) = K;

    for (int i = 0; i < 4; ++i) {
        Matrix34d Rt1;
        Rt1 << possible_Rs[i], possible_ts[i];
        Matrix34d P1 = K * Rt1;

        int count_valid = 0;
        std::vector<Eigen::Vector3d> tri_points;
        std::vector<int> good_indices;
        for (int j = 0; j < matched_pts0.size(); ++j) {
            Eigen::Vector3d X = triangulatePoint(matched_pts0[j], P0, matched_pts1[j], P1);

            double z0 = X.z();
            double z1 = (Rt1 * X.homogeneous()).z();
            if (z0 > 0 && z0 < dmax && z1 > 0 && z1 < dmax) {
                good_indices.push_back(j);
                count_valid++;
                tri_points.push_back(X);
            }
        }

        if (count_valid > max_count_valid) {
            max_count_valid = count_valid;
            best_solution_idx = i;
            triangulated_points = std::move(tri_points);
            best_solution_indices = std::move(good_indices);
        }
    }

    std::cout << "Best solution has triangulated " << max_count_valid << "/" << matches.size() << " points in front of the camera.\n";

    Eigen::Matrix<double, 3, 4> Rt;
    Rt.block<3 ,3>(0, 0) = possible_Rs[best_solution_idx];
    Rt.col(3) = possible_ts[best_solution_idx];


    // Filter matches
    std::vector<cv::DMatch> filtered_matches(best_solution_indices.size());
    int a = 0;
    for (auto i : best_solution_indices) {
        filtered_matches[a++] = matches[i];
    }
    matches = std::move(filtered_matches);


    return Rt;    
}
