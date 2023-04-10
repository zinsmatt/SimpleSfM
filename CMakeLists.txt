cmake_minimum_required(VERSION 3.14)
project(SimpleSFM)

set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Eigen3 REQUIRED)

# OpenCV
find_package( OpenCV REQUIRED )

# G2O
find_package(g2o REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
main.cxx
camera.cxx
epipolar.cxx
features2d.cxx
frame.cxx
io.cxx
map.cxx
matching.cxx
optimize.cxx
point3d.cxx
triangulation.cxx
utils.cxx
)

target_link_libraries(${PROJECT_NAME}
${EIGEN3_LIBS}
${OpenCV_LIBS}
g2o::core g2o::stuff g2o::types_sba g2o::solver_csparse g2o::csparse_extension
${CSPARSE_LIBRARY}
)


