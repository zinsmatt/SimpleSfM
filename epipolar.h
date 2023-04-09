#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "features2d.h"
#include "utils.h"


Eigen::Matrix3d computeEssentialMatrix(const std::vector<cv::Point2d>& pts0, const std::vector<cv::Point2d>& pts1,
                                       std::vector<cv::DMatch>& matches, const Eigen::Matrix3d& K);


Matrix34d computeRtFromEssential(const Eigen::Matrix3d& E, const std::vector<Vector2d>& pts0,
                                 const std::vector<Vector2d>& pts1, std::vector<cv::DMatch>& matches,
                                 const Eigen::Matrix3d& K, std::vector<Eigen::Vector3d>& triangulated_points,
                                 double dmax);