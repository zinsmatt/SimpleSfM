#include "io.h"

#include <fstream>


void saveImageDescriptors(const std::string filename, const std::vector<ImageDescriptor::Ptr>& descriptors) {
    std::ofstream of(filename);
    of << descriptors.size() << "\n";
    for (auto desc : descriptors) {
        of << desc->serialize();
        of << "eol\n";
    }
    of.close();
}

bool loadImageDescriptors(const std::string filename, std::vector<ImageDescriptor::Ptr>& out_descriptors) {
    std::ifstream fin(filename);
    if (!fin.is_open())
        return false;
    
    std::string line;
    int n;
    fin >> n;
    std::getline(fin, line);
    int count = 0;
    for (int i = 0; i < n; ++i) {
        std::stringstream ss;
        while (std::getline(fin, line)) {
            if (line == "eol")
                break;
            ss << line << "\n";
        }
        out_descriptors.push_back(ImageDescriptor::deserialize(ss.str()));
        ++count;
    }
    fin.close();

    std::cout << "Loaded " << count << " new image descriptors.\n";
    return true;
}




std::string serializeTriaxeOBJ(const Eigen::Matrix3d &orientation, const Eigen::Vector3d &position,
                               double size, double nPoints)
{
    Eigen::Matrix<double, 3, Eigen::Dynamic> pts = Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, 3*nPoints);
    double step = size / nPoints;
    for (int i = 0; i < nPoints; ++i) {
        pts(0, i) = i * step;
        pts(1, nPoints+i) = i * step;
        pts(2, 2*nPoints+i) = i * step;
    }

    auto triaxe = (orientation * pts).colwise() + position;
    
    std::stringstream ss;
    for (int i = 0; i < 3 * nPoints; ++i) {
        ss << "v " << triaxe(0, i) << " " << triaxe(1, i) << " " << triaxe(2, i);
        int color[3] = {0, 0, 0};
        if (i < nPoints) {
            color[0] = 255;
        } else if (i < 2*nPoints) {
            color[1] = 255;
        } else {
            color[2] = 255;
        }
        ss << " " << color[0] << " " << color[1] << " " << color[2];
        ss << "\n";
    }
    return ss.str();
}


void saveCamerasOBJ(const std::string& filename, const std::vector<Frame::Ptr> &frames, double size, int n_points) {
    std::ofstream fout(filename);
    for (auto frame : frames) {
        fout << frame->serializeToOBJ(size, n_points);
    }
    fout.close();
}

void writeTriaxeOBJ(const std::string& filename, const std::vector<Eigen::Matrix3d> &orientations,
                    const std::vector<Eigen::Vector3d> &positions, double size, double nPoints)
{
    std::ofstream fout(filename);
    for (int i = 0; i < orientations.size(); ++i) {
        fout << serializeTriaxeOBJ(orientations[i], positions[i], size, nPoints);
    }
    fout.close();
}


void writeOBJ(const std::string& filename, const std::vector<Eigen::Vector3d>& points) {
    std::ofstream fout(filename);
    for (auto& p : points) {
        fout << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    fout.close();
}