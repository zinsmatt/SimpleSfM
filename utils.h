#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using Matrix34d = Eigen::Matrix<double, 3, 4>;

inline Eigen::Vector2d cvpoint2eigen(const cv::Point2d& pt) {
    return Eigen::Vector2d(pt.x, pt.y);    
}


template <class T, int R, int C>
Eigen::Matrix<T, R, C> cv2eigen(cv::Mat M) {
    Eigen::Matrix<T, R, C> m;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            m(i, j) = M.at<T>(i, j);
    return m;
}


template <class T, int R, int C>
cv::Mat eigen2cv(const Eigen::Matrix<T, R, C>& M) {
    cv::Mat m(R, C, CV_64F);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            m.at<double>(i, j) = M(i, j);
    return m;
}

