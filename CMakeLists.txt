cmake_minimum_required(VERSION 3.14)
project(SimpleSFM)

set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -pthread")

find_package(Eigen3 REQUIRED)


find_package( OpenCV REQUIRED )


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
point3d.cxx
triangulation.cxx
utils.cxx
)

target_link_libraries(${PROJECT_NAME}
${EIGEN3_LIBS}
${OpenCV_LIBS}
)


