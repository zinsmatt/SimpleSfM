#pragma once

#include "utils.h"

struct Intrinsics
{
    Eigen::Matrix3d K;

    Intrinsics(const Matrix3d& calib) : K(calib) {}
};