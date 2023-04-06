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



std::vector<cv::DMatch> RobustMatcher::robustMatch(ImageDescriptor::Ptr desc1, ImageDescriptor::Ptr desc2)
{
    std::vector<std::vector<cv::DMatch>> matches12, matches21;
    cv::BFMatcher matcher(cv::NORM_HAMMING, false);

    matcher.knnMatch(desc1->descriptors, desc2->descriptors, matches12, 2);
    matcher.knnMatch(desc2->descriptors, desc1->descriptors, matches21, 2);

    ratioTest(matches12);
    std::cout << matches12.size() << " matches after ratio test.\n";

    ratioTest(matches21);
    std::cout << matches21.size() << " matches after ratio test.\n";

    
    std::vector<cv::DMatch> good_matches;
    symmetryTest(matches12, matches21, good_matches);
    std::cout << good_matches.size() << " good symmetric matches.\n";
    return good_matches;
}