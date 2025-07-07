cmake_minimum_required(VERSION 3.15)
project(KCF_YOLO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get the UI code
include(FetchContent)
FetchContent_Declare(
        ftxui
        GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
        GIT_TAG main
)
FetchContent_MakeAvailable(ftxui)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Set ONNX Runtime root directory
set(ONNXRUNTIME_DIR "onnxruntime-osx-x86_64-1.20.1")

# set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime")
# if(NOT ONNXRUNTIME_DIR)
#     message(FATAL_ERROR "Please set ONNXRUNTIME_DIR: cmake -DONNXRUNTIME_DIR=/path/to/onnxruntime ...")
# endif()

# Manually include the 'onnxruntime' subfolder in the include path
include_directories(${ONNXRUNTIME_DIR}/include)

# Add ONNX Runtime library path
link_directories(${ONNXRUNTIME_DIR}/lib)

# Add source and include directories
file(GLOB_RECURSE SOURCES "src/*.cpp")
include_directories(include)

# Add executable
add_executable(KCF_YOLO_Tracker ${SOURCES})

# Link libraries
target_link_libraries(KCF_YOLO_Tracker
        ${OpenCV_LIBS}
        onnxruntime
        ftxui::component
)

message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}")
