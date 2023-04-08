#pragma once

#include <Eigen/Dense>

#include <vector>

Eigen::Vector3d triangulatePoint(const Eigen::Vector2d& p1, const Eigen::Matrix<double, 3, 4>& P1,
                                 const Eigen::Vector2d& p2, const Eigen::Matrix<double, 3, 4>& P2);

std::vector<Eigen::Vector3d> triangulatePoints(const std::vector<Eigen::Vector2d>& pts1, const Eigen::Matrix<double, 3, 4>& P1,
                                               const std::vector<Eigen::Vector2d>& pts2, const Eigen::Matrix<double, 3, 4>& P2);