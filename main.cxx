#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <memory>


using namespace std;

struct ImageDescriptor
{
    typedef std::shared_ptr<ImageDescriptor> Ptr;
    static Ptr create() {
        return std::make_shared<ImageDescriptor>();
    }
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;

    std::string serialize() const {
        stringstream ss;
        ss << keypoints.size() << "\n";
        for (auto& kp : keypoints) {
            ss << kp.pt.x << " " << kp.pt.y << " " << kp.size << "\n";
        }
        ss << descriptors.cols << "\n";
        for (int i = 0; i < descriptors.rows; ++i) {
            for (int j = 0; j < descriptors.cols; ++j) {
                ss << static_cast<int>(descriptors.at<uint8_t>(i, j)) << " ";
            }
            ss << "\n";
        }
        return ss.str();
    }

    static Ptr deserialize(const std::string& serial) {
        auto obj = std::make_shared<ImageDescriptor>();
        std::istringstream iss(serial);
        int n;
        iss >> n;
        for (int i = 0; i < n; ++i) {
            float x, y, s;
            iss >> x >> y >> s;
            obj->keypoints.push_back(cv::KeyPoint(x, y, s));
        }
        int m;
        iss >> m;
        obj->descriptors.create(n, m, CV_8U);
        int c;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                iss >> c;
                obj->descriptors.at<uint8_t>(i, j) = static_cast<uint8_t>(c);
            }
        }

        return obj;
    }

};


void saveImageDescriptors(const std::string filename, const std::vector<ImageDescriptor::Ptr>& descriptors) {
    std::ofstream of(filename);
    of << descriptors.size() << "\n";
    for (auto desc : descriptors) {
        of << desc->serialize();
        of << "eol\n";
    }
    of.close();
}

bool loadImageDescriptors(const std::string filename, std::vector<ImageDescriptor::Ptr>& out_descriptors) {
    std::ifstream fin(filename);
    if (!fin.is_open())
        return false;
    
    std::string line;
    int n;
    fin >> n;
    std::getline(fin, line);
    int count = 0;
    for (int i = 0; i < n; ++i) {
        std::stringstream ss;
        while (std::getline(fin, line)) {
            if (line == "eol")
                break;
            ss << line << "\n";
        }
        out_descriptors.push_back(ImageDescriptor::deserialize(ss.str()));
        ++count;
    }
    fin.close();

    std::cout << "Loaded " << count << " new image descriptors.\n";
    return true;
}


ImageDescriptor::Ptr computeImageDescriptors(cv::Mat img) {
    ImageDescriptor::Ptr desc = ImageDescriptor::create();
    double nKeypoints = 2000;
    double scaleFactor = 1.5;
    cv::Ptr<cv::ORB> detector = cv::ORB::create(nKeypoints, scaleFactor);
    detector->detectAndCompute(img, cv::Mat(), desc->keypoints, desc->descriptors);
    return desc;

}


class RobustMatcher
{
    double ratiotest_;

    void ratioTest(std::vector<std::vector<cv::DMatch>> &matches) {
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

    void symmetryTest(const std::vector<std::vector<cv::DMatch>>& matches12, 
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


public:

    RobustMatcher(double ratiotest) : ratiotest_(ratiotest) {}


    std::vector<cv::DMatch> robustMatch(ImageDescriptor::Ptr desc1, ImageDescriptor::Ptr desc2)
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
};


int main() {

    std::vector<cv::Mat> images;
    std::vector<ImageDescriptor::Ptr> imageDescriptors;

    for (int i = 0; i <= 7; ++i) {
        char buffer[1024];
        std::sprintf(buffer, "../data/herzjesu/%04d.png", i);
        std::string name(buffer);
        cv::Mat img_init = cv::imread(name);
        cv::Mat img;
        cv::resize(img_init, img, cv::Size(img_init.size().width * 0.3, img_init.size().height * 0.3));
        images.push_back(img);


        // auto desc = computeImageDescriptors(img);
        // std::cout << desc->descriptors.type() << "\n";
        // imageDescriptors.push_back(desc);
        // std::cout << desc->keypoints.size() << " poins detected\n";

        // cv::Mat img_keypoints;
        // cv::drawKeypoints(img, desc->keypoints, img_keypoints);
        // cv::imshow("keypoints", img_keypoints);
        // cv::waitKey();
    }

    // saveImageDescriptors("image_descriptors.txt", imageDescriptors);

    loadImageDescriptors("image_descriptors.txt", imageDescriptors);

    for (int i = 0; i < imageDescriptors.size(); ++i) {
        auto d = imageDescriptors[i];
        auto d2 = imageDescriptors[i];
        std::cout << d->keypoints.size() << "\n";
        std::cout << d2->keypoints.size() << "\n";

        for (int j = 0; j < d->keypoints.size(); ++j) {
            auto kp = d->keypoints[j];
            auto kp2 = d2->keypoints[j];
            if (std::abs(kp.pt.x - kp2.pt.x) > 1.0e-3 ||
                std::abs(kp.pt.y - kp2.pt.y) > 1.0e-3 ||
                std::abs(kp.size - kp2.size) > 1.0e-3)
                std::cout << "ERROR keypoints\n";
        }


        std::cout << d->descriptors.size << "\n";
        std::cout << d2->descriptors.size << "\n";
        for (int ii = 0; ii < d->descriptors.rows; ++ii) {
            for (int j = 0; j < d->descriptors.cols; ++j) {
                auto a = d->descriptors.at<uint8_t>(ii, j);
                auto a2 = d2->descriptors.at<uint8_t>(ii, j);
                if (a != a2)
                    std::cout << ii << " " << j <<  " : ERROR descriptor\n";
            }
        }

    }

    for (int i = 0; i <= 7; ++i)
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
    }




    

    return 0;
}