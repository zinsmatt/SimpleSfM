#pragma once


#include "frame.h"

#include "map.h"

void runBundleAdjustment(const std::vector<Frame::Ptr> &frames, Map& map, int init_cam, int n_iterations);