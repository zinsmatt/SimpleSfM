#include "optimize.h"

#include <Eigen/Dense>
#include "map.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/core/robust_kernel_impl.h"

#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/sba/edge_project_xyz.h>


void runBundleAdjustment(const std::vector<Frame::Ptr> &frames, Map& map, int init_cam, int n_iterations) {
    
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = std::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
          std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));

  optimizer.setAlgorithm(solver);


    std::vector<g2o::VertexSE3Expmap*> vertices_cameras;
    for (int i = 0; i < frames.size(); ++i) {
        Frame::Ptr frame = frames[i];

        const auto& Rt = frame->getRt();

        Eigen::Quaterniond q(Rt.block<3, 3>(0, 0));
        g2o::SE3Quat pose(q, Rt.col(3));

        g2o::VertexSE3Expmap *v = new g2o::VertexSE3Expmap();

        v->setId(i);
        v->setEstimate(pose);
        // std::cout << pose.to_homogeneous_matrix() << "\n";

        if (i == init_cam) {
            v->setFixed(true);
        }

        optimizer.addVertex(v);
        vertices_cameras.push_back(v);
    }

    std::unordered_map<Point3d*, int> points_to_index;
    
    auto const& K = frames[0]->getK();

    std::vector<g2o::VertexPointXYZ*> vertices_points;
    for (int i = 0; i < map.size(); ++i) {
        g2o::VertexPointXYZ* vp = new g2o::VertexPointXYZ();
        vp->setId(vertices_cameras.size() + i);
        vp->setMarginalized(true);
        auto pt = map.getPoint(i);
        // std::cout << pt->getPos().transpose() << "\n";
        points_to_index[pt.get()] = i;
        vp->setEstimate(pt->getPos());
        optimizer.addVertex(vp);
        vertices_points.push_back(vp);
    }

    for (int i = 0; i < frames.size(); ++i) {
        auto& feats = frames[i]->getFeatures();
        for (auto f : feats) {
            auto* pt3d = f->getPointRef();
            if (pt3d) {
                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                e->setVertex(0, vertices_points[points_to_index[pt3d]]);
                e->setVertex(1, vertices_cameras[i]);
                e->setMeasurement(f->getPos());
                e->information() = Matrix2d::Identity();
                e->fx = K(0, 0);
                e->fy = K(1, 1);
                e->cx = K(0, 2);
                e->cy = K(1, 2);

                g2o::RobustKernel *rk = new g2o::RobustKernelHuber();
                e->setRobustKernel(rk);

                optimizer.addEdge(e);
            }
        }
    }

    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(n_iterations);


    for (int i = 0; i < vertices_points.size(); ++i) {
        map.getPoint(i)->setPos(vertices_points[i]->estimate());       
    }

    for (int i = 0; i < vertices_cameras.size(); ++i) {
        g2o::SE3Quat pose = vertices_cameras[i]->estimate();
        frames[i]->setRt(pose.to_homogeneous_matrix().block<3, 4>(0, 0));
    }
}