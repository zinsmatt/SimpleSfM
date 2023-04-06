#pragma once

#include "features2d.h"

void saveImageDescriptors(const std::string filename, const std::vector<ImageDescriptor::Ptr>& descriptors);

bool loadImageDescriptors(const std::string filename, std::vector<ImageDescriptor::Ptr>& out_descriptors);
