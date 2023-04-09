#include "triangulation.h"

#include "point3d.h"


Eigen::Vector3d triangulatePoint(const Eigen::Vector2d& p1, const Eigen::Matrix<double, 3, 4>& P1,
                                 const Eigen::Vector2d& p2, const Eigen::Matrix<double, 3, 4>& P2)
{
    // Eigen::Matrix4d A;
    // A.row(0)  = p1.x() * P1.row(2) - P1.row(0);
    // A.row(1)  = p1.y() * P1.row(2) - P1.row(1);
    // A.row(2)  = p2.x() * P2.row(2) - P2.row(0);
    // A.row(3)  = p2.y() * P2.row(2) - P2.row(1);

    // Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);

    // Eigen::Vector4d X = svd.matrixV().col(3);
    // return X.head(3) / X[3];




    Eigen::Matrix4d A(4, 4);
    A.row(0) = p1[0] * P1.row(2) - P1.row(0);
    A.row(1) = p1[1] * P1.row(2) - P1.row(1);
    A.row(2) = p2[0] * P2.row(2) - P2.row(0);
    A.row(3) = p2[1] * P2.row(2) - P2.row(1);
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);
    Eigen::Vector3d center = X.head(3) / X[3];
    return center;

}


std::vector<Eigen::Vector3d> triangulatePoints(const std::vector<Eigen::Vector2d>& pts1, const Eigen::Matrix<double, 3, 4>& P1,
                                               const std::vector<Eigen::Vector2d>& pts2, const Eigen::Matrix<double, 3, 4>& P2)
{
    std::vector<Eigen::Vector3d> out(pts1.size());
    for (int i = 0; i < pts1.size(); ++i) {
        out[i] = triangulatePoint(pts1[i], P1 ,pts2[i], P2);
    }
    return out;
}


void triangulateFeatures(Frame::Ptr frame0, Frame::Ptr frame1, MatchList::Ptr matches, Map& map, double dmax)
{
    std::cout << "before get P Rt" << std::endl;

    const auto& P0 = frame0->getP();
    const auto& P1 = frame1->getP();
    const auto& Rt0 = frame0->getRt();
    const auto& Rt1 = frame1->getRt();
    std::cout << "before getfeatures" << std::endl;
    auto& features0 = frame0->getFeatures();
    auto& features1 = frame1->getFeatures();
    std::cout << "Start reconstruction" << std::endl;
    for (auto &m : *matches) {
        auto& f0 = features0[m.queryIdx];
        auto& f1 = features1[m.trainIdx];

        auto X = triangulatePoint(f0->getPos(), P0, f1->getPos(), P1);

        double z0 = (Rt0 * X.homogeneous()).z();
        double z1 = (Rt1 * X.homogeneous()).z();
        if (z0 > 0 && z0 < dmax && z1 > 0 && z1 < dmax) {
            auto pointPtr = map.addPoint(X);
            f0->setPointRef(pointPtr);
            f1->setPointRef(pointPtr);
        }
    }
}