#include "matching.h"


void RobustMatcher::ratioTest(std::vector<std::vector<cv::DMatch>> &matches)
{
    for (int i = 0; i < matches.size(); ++i) {
        if (matches[i].size() > 1) {
            if (matches[i][0].distance > matches[i][1].distance * ratiotest_) {
                matches[i].clear();
            }

        } else {
            matches[i].clear();
        }
    }
}

void RobustMatcher::symmetryTest(const std::vector<std::vector<cv::DMatch>>& matches12, 
                    const std::vector<std::vector<cv::DMatch>>& matches21,
                    std::vector<cv::DMatch>& symMatches)
{
    for (int i = 0; i < matches12.size(); ++i) {
        if (matches12[i].size() < 2)
            continue;

        for (int j = 0; j < matches21.size(); ++j) {
            if (matches21[j].size() < 2)
                continue;
            
            if (matches12[i][0].trainIdx == matches21[j][0].queryIdx &&
                matches12[i][0].queryIdx == matches21[j][0].trainIdx)
                {
                    symMatches.push_back(matches12[i][0]);
                    break;
                }
        }
    }
}



MatchList::Ptr RobustMatcher::robustMatch(Frame::Ptr frame1, Frame::Ptr frame2)
{
    std::vector<std::vector<cv::DMatch>> matches12, matches21;
    // cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    cv::BFMatcher matcher(cv::NORM_L2, false);

    matcher.knnMatch(frame1->getDescriptors(), frame2->getDescriptors(), matches12, 2);
    matcher.knnMatch(frame2->getDescriptors(), frame1->getDescriptors(), matches21, 2);

    ratioTest(matches12);
    // std::cout << matches12.size() << " matches after ratio test.\n";

    ratioTest(matches21);
    // std::cout << matches21.size() << " matches after ratio test.\n";

    
    auto good_matches = MatchList::create();
    symmetryTest(matches12, matches21, *good_matches);
    std::cout << good_matches->size() << " good symmetric matches.\n";
    return good_matches;
}