#pragma once

#include <Eigen/Dense>


Eigen::Vector3d triangulatePoint(const Eigen::Vector2d& p1, const Eigen::Matrix<double, 3, 4>& P1,
                                 const Eigen::Vector2d& p2, const Eigen::Matrix<double, 3, 4>& P2);