#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "features2d.h"
#include "utils.h"


Eigen::Matrix3d computeEssentialMatrix(ImageDescriptor::Ptr desc1, ImageDescriptor::Ptr desc2,
                                       std::vector<cv::DMatch>& matches, const Eigen::Matrix3d& K);


Matrix34d computeRtFromEssential(const Eigen::Matrix3d& E, ImageDescriptor::Ptr &desc1,
                                 ImageDescriptor::Ptr desc2, std::vector<cv::DMatch>& matches,
                                 const Eigen::Matrix3d& K, std::vector<Eigen::Vector3d>& triangulatedPoints,
                                 double dmax);