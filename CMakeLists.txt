cmake_minimum_required(VERSION 3.15)
project(KCF_YOLO_Tracker)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Set ONNX Runtime root directory
set(ONNXRUNTIME_DIR "/usr/local/Cellar/onnxruntime/1.21.1")

# Manually include the 'onnxruntime' subfolder in the include path
include_directories(${ONNXRUNTIME_DIR}/include/onnxruntime)

# Add source and include directories
file(GLOB_RECURSE SOURCES "src/*.cpp")
include_directories(include)

# Add executable
add_executable(KCF_YOLO_Tracker ${SOURCES})

# Link libraries
target_link_libraries(KCF_YOLO_Tracker
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib
)

message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}")
