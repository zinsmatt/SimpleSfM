#pragma once

#include <Eigen/Dense>

#include "utils.h"

class Feature2d;

class Point3d
{
    Vector3d pos_;
    std::set<Feature2d*> observations_;

public:
    typedef std::shared_ptr<Point3d> Ptr;
    static Ptr create(const Vector3d& pos) {
        return std::make_shared<Point3d>(pos);
    }

    Point3d(const Vector3d& pos) : pos_(pos) {}

    const Eigen::Vector3d& getPos() const {
        return pos_;
    }

    void setPos(const Vector3d& p) {
        pos_ = p;
    }

    void addObservation(Feature2d* obs) {
        observations_.insert(obs);
    }

    const std::set<Feature2d*>& getObservations() {
        return observations_;
    }

    
};