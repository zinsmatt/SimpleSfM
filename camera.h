#pragma once

#include "utils.h"

struct Intrinsics
{
    typedef std::shared_ptr<Intrinsics> Ptr;
    static Ptr create(const Matrix3d& calib) {
        return std::make_shared<Intrinsics>(calib);
    }
    
    Eigen::Matrix3d K;

    Intrinsics(const Matrix3d& calib) : K(calib) {}
};