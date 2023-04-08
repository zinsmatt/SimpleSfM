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