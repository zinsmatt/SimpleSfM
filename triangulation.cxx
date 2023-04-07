#include "triangulation.h"

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
