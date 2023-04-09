#pragma once

#include <opencv2/core.hpp>
#include "features2d.h"
#include "frame.h"


class RobustMatcher
{
    double ratiotest_;

    void ratioTest(std::vector<std::vector<cv::DMatch>> &matches);

    void symmetryTest(const std::vector<std::vector<cv::DMatch>>& matches12, 
                      const std::vector<std::vector<cv::DMatch>>& matches21,
                      std::vector<cv::DMatch>& symMatches);

public:

    RobustMatcher(double ratiotest) : ratiotest_(ratiotest) {}

    std::vector<cv::DMatch> robustMatch(Frame::Ptr frame1, Frame::Ptr frame2);
};