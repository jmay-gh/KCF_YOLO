# WildDrone Multi-Object Tracker

This is a C++ multi-object tracking system that combines **YOLO object detection** with **KCF tracking**, using multiple data association strategies (Hungarian, greedy, EMD) and various similarity metrics (IoU, HOG, Euclidean, 3D).

Developed to be used on drones, for tracking wild animals in real-time in African game parks.

## 🚧 Still Under Development

## Features

- 🎯 YOLO-based object detection
- 🚀 KCF tracker per-object
- 🔁 Data association:
    - Greedy matching
    - Hungarian algorithm
    - Earth Mover’s Distance (EMD)
- 📐 Similarity metrics:
    - Euclidean distance
    - Intersection over Union (IoU)
    - Histogram of Oriented Gradients (HOG)
    - 3D spatial distance + area
- 🧪 UI to select modes and configs
- 🎥 Save results or visualize tracking live

## Build

### Dependencies
- C++20
- OpenCV (>=4.5)
- YOLOv8+ ONNX variant
- Depth Anything v2
- CMake (>=3.12)

### Build

```bash
git clone https://github.com/jmay-gh/multi-object-tracker.git
cd multi-object-tracker
mkdir build && cd build
cmake ..
make -j$(nproc)
