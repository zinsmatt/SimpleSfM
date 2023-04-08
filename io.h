#pragma once

#include "features2d.h"

void saveImageDescriptors(const std::string filename, const std::vector<ImageDescriptor::Ptr>& descriptors);

bool loadImageDescriptors(const std::string filename, std::vector<ImageDescriptor::Ptr>& out_descriptors);


std::string serializeTriaxeOBJ(const Eigen::Matrix3d &orientation, const Eigen::Vector3d &position,
                               double size=1.0, double nPoints=100);

void writeTriaxeOBJ(const std::string& filename, const std::vector<Eigen::Matrix3d> &orientations,
                    const std::vector<Eigen::Vector3d> &positions, double size=1.0, double nPoints=100);


void writeOBJ(const std::string& filename, const std::vector<Eigen::Vector3d>& points);
