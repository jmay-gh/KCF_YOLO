#include "../include/tracking/TrackerManager.h"
#include "seg/YOLO11Seg.hpp"
#include "depth/depth_anything.hpp"
#include "../include/UserInterface.h"
#include "../include/testing/TestingHandler.h"

#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;
using namespace testing;

int main(void) {

    string filePath = "../img/horse_7/images.txt";
    string groundTruthPath = "../img/deer_7/ground_truth.txt";

    // Set up and run user interface
    TrackerConfig config;
    UserInterface ui;
    ui.run(config);

    // Set up YOLO detector
    const string labelsPath = "../models/coco.names";
    const string modelPath = "../models/yolo11s-seg.onnx";
    YOLOv11SegDetector detector(modelPath, labelsPath, true);

    // Set up depth detector
    const string depthModelPath = "../models/depth_anything_v2_vits.onnx";
    DepthAnything depthEstimator(depthModelPath, false);
    cv::Mat depthMap;
    cv::Mat depthColor;

    // Set up image frames
    ifstream listFramesFile(filePath);
    string frameName;
    Mat frame;
    int frameIdx = 0;

    // Set up tracker manager
    vector<string> classNames = ::utils::getClassNames(labelsPath);
    TrackerManager trackerManager(config, classNames, frame);

    // Run tests
    if (config.testing == TrackerConfig::TestingType::MOT_ACCURACY) {
        TestingHandler testingHandler(config);
        // Run MOT accuracy tests
        testingHandler.runMOTAccuracyTests(groundTruthPath, frame);
        return 0;
    }

    // Set up video writer
    string firstFrameName;
    getline(listFramesFile, firstFrameName);
    Mat firstFrame = imread(firstFrameName, IMREAD_COLOR);
    int frameWidth = firstFrame.cols;
    int frameHeight = firstFrame.rows;
    double fps = 20.0;
    string outputVideoPath = "../output/tracking_output.avi";
    VideoWriter videoWriter;
    videoWriter.open(outputVideoPath, VideoWriter::fourcc('M','J','P','G'),
                     fps, Size(frameWidth, frameHeight));
    listFramesFile.close();
    listFramesFile.open(filePath);

    // Iterate image frames
    while (getline(listFramesFile, frameName)) {
        frame = imread(frameName, cv::IMREAD_COLOR);
        if (frame.empty()) continue;
        // Redetect every 8 frames
        if (frameIdx % 8 == 0) {
            depthMap = depthEstimator.predict(frame.clone());
            auto detections = detector.segment(frame);
            // Remove low confidence detections
            for (auto it = detections.begin(); it != detections.end(); ) {
                if (it->conf <= 0.3) it = detections.erase(it);
                else ++it;
            }
            trackerManager.updateTrackersWithDetections(frame, detections, depthMap);
        }
        else {
            trackerManager.updateTrackers(frame);
        }


        // Visualisation settings
        if (true) { trackerManager.drawTrackers(frame); }
        else {
            // TODO - Set up depth Map visualisation
            cv::Mat depthVis;
            cv::normalize(depthMap, depthVis, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(depthVis, depthColor, cv::COLORMAP_JET);
            cv::Mat blended;
            cv::addWeighted(frame, 0.4, depthColor, 0.6, 0.0, blended);
            trackerManager.drawTrackers(blended);
        }

        // Output settings
        if (config.output == TrackerConfig::SHOW || config.output == TrackerConfig::BOTH) {
            imshow("Tracking", frame);
            cv::waitKey(1);
        }
        if (config.output == TrackerConfig::SAVE) videoWriter.write(frame);
        ++frameIdx;
    }

    for (auto& tracker : trackerManager.trackers) {
        cout << "Tracker " << tracker.trackerNum << endl;
    }

    return 0;
}
