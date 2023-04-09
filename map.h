#pragma once

#include "point3d.h"

#include "io.h"
class Map
{
    std::vector<Point3d::Ptr> points_;

public:
    Point3d* addPoint(const Vector3d& pos) {
        points_.push_back(Point3d::create(pos));
        return points_.back().get();
    }

    void saveMap(const std::string filename) {
        std::vector<Vector3d> pts(points_.size());
        for (int i = 0; i < pts.size(); ++i) {
            pts[i] = points_[i]->getPos();
        }
        writeOBJ(filename, pts);
    }
};