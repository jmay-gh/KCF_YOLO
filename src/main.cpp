#include "../include/tracking/TrackerManager.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

    // Set up YOLO detector
    const string labelsPath = "../models/coco.names";
    const string modelPath = "../models/yolo11m.onnx";
    YOLO11Detector detector(modelPath, labelsPath, true);

    // Set up tracker manager
    vector<string> classNames = ::utils::getClassNames(labelsPath);
    TrackerConfig config = TrackerConfig::parseArgs(argc, argv);
    TrackerManager trackerManager(config, classNames);

    // Set up image frames
    ifstream listFramesFile("../img/zebra_1/images.txt");
    string frameName;
    Mat frame;
    int frameIdx = 0;

    // Iterate image frames
    while (std::getline(listFramesFile, frameName)) {
        frame = cv::imread(frameName, cv::IMREAD_COLOR);
        if (frame.empty()) continue;
        if (frameIdx % 8 == 0) {
            auto detections = detector.detect(frame);
            trackerManager.updateTrackers(frame, detections, frameIdx);
        }
        else {
            trackerManager.updateTrackers(frame, {}, frameIdx);
        }

        trackerManager.drawTrackers(frame);

        if (!config.SILENT) {
            imshow("Tracking", frame);
            cv::waitKey(1);
        }
        ++frameIdx;
    }
    return 0;
}
