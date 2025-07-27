#pragma once

#include "tracking/TrackerManager.h"
#include "depth/depth_anything.hpp"
#include "userSetup/UserInterface.h"

#include <opencv2/videoio.hpp>
#include <filesystem>

using namespace std;
using namespace trackingUtils;
namespace fs = std::filesystem;

void runTrackingOnDataset(const std::string& filePath,
                          const std::string& groundTruthPath,
                          const UserConfig& config,
                          YOLOv11SegDetector& detector,
                          DepthAnything& depthEstimator,
                          const std::vector<std::string>& classNames,
                          int datasetIdx);

int main() {
    // Run UI once to get config
    UserConfig config;
    UserInterface ui;
    ui.run(config);


    // Set up YOLO detector
    const string labelsPath = "../models/coco.names";
    const string modelPath = "../models/yolo11s-seg.onnx";
    YOLOv11SegDetector detector(modelPath, labelsPath, true);

    // Set up depth model
    const string depthModelPath = "../models/depth_anything_v2_vits.onnx";
    DepthAnything depthEstimator(depthModelPath, false);

    // Load class names
    vector<string> classNames = ::utils::getClassNames(labelsPath);

    // List of datasets to process
    vector<pair<string, string>> datasets = {
//            {"../img/horse_1/images.txt", "../img/horse_1/horse_1_gt.txt"},
//            {"../img/horse_2/images.txt", "../img/horse_2/horse_2_gt.txt"},
//            {"../img/horse_3/images.txt", "../img/horse_3/horse_3_gt.txt"},
//            {"../img/horse_4/images.txt", "../img/horse_4/horse_4_gt.txt"},
//            {"../img/horse_5/images.txt", "../img/horse_5/horse_5_gt.txt"},
//            {"../img/horse_6/images.txt", "../img/horse_6/horse_6_gt.txt"},
//            {"../img/horse_7/images.txt", "../img/horse_7/horse_7_gt.txt"},
//            {"../img/deer_1/images.txt", "../img/deer_1/deer_1_gt.txt"},
//            {"../img/deer_2/images.txt", "../img/deer_2/deer_2_gt.txt"},
//            {"../img/deer_3/images.txt", "../img/deer_3/deer_3_gt.txt"},
//            {"../img/deer_4/images.txt", "../img/deer_4/deer_4_gt.txt"},
//            {"../img/deer_5/images.txt", "../img/deer_5/deer_5_gt.txt"},
//            {"../img/deer_6/images.txt", "../img/deer_6/deer_6_gt.txt"},
//            {"../img/deer_7/images.txt", "../img/deer_7/deer_7_gt.txt"},
            {"../img/zebra_1/images.txt", "../img/zebra_1/zebra_1_gt.txt"},
//            {"../img/zebra_2/images.txt", "../img/zebra_2/zebra_2_gt.txt"},
//            {"../img/zebra_3/images.txt", "../img/zebra_3/zebra_3_gt.txt"},
//            {"../img/zebra_4/images.txt", "../img/zebra_4/zebra_4_gt.txt"},
//            {"../img/zebra_5/images.txt", "../img/zebra_5/zebra_5_gt.txt"},
    };

    // Process each dataset
    for (size_t i = 0; i < datasets.size(); ++i) {
        const auto& [imgList, gtPath] = datasets[i];
        runTrackingOnDataset(imgList, gtPath, config, detector,
                             depthEstimator, classNames, static_cast<int>(i));
    }
    return 0;
}


void runTrackingOnDataset(const std::string& filePath,
                          const std::string& groundTruthPath,
                          const UserConfig& config,
                          YOLOv11SegDetector& detector,
                          DepthAnything& depthEstimator,
                          const vector<string>& classNames,
                          int datasetIdx) {

    ifstream listFramesFile(filePath);
    if (!listFramesFile.is_open()) {
        cerr << "Could not open image list: " << filePath << endl;
        return;
    }

    string frameName;
    getline(listFramesFile, frameName);
    cv::Mat firstFrame = imread(frameName, cv::IMREAD_COLOR);
    if (firstFrame.empty()) {
        cerr << "Could not read first frame from: " << frameName << endl;
        return;
    }

    int frameWidth = firstFrame.cols;
    int frameHeight = firstFrame.rows;
    double fps = 20.0;

    // Reset file
    listFramesFile.clear();
    listFramesFile.seekg(0);

    string animalType;
    if (filePath.find("horse") != string::npos) animalType = "horse";
    else if (filePath.find("zebra") != string::npos) animalType = "zebra";
    else if (filePath.find("deer") != string::npos) animalType = "deer";

    string outputVideoPath = "../results/output_" + to_string(datasetIdx) + ".avi";
    cv::VideoWriter videoWriter(outputVideoPath, cv::VideoWriter::fourcc('M','J','P','G'),
                            fps, cv::Size(frameWidth, frameHeight));

    string resultsPath = "../results/" + animalType + "_" + to_string(datasetIdx+1) + ".txt";
    ofstream outputFile(resultsPath);
    if (!outputFile.is_open()) {
        cerr << "Could not write to: " << resultsPath << endl;
        return;
    }

    // Tracker manager
    cv::Mat frame, depthMap, depthColor;
    TrackerManager trackerManager(config, frame);

    int frameIdx = 0;

    while (getline(listFramesFile, frameName)) {
        frame = imread(frameName, cv::IMREAD_COLOR);

        if (frame.empty()) continue;

        if (frameIdx % 8 == 0) {
            depthMap = depthEstimator.predict(frame);

            auto detections = detector.segment(frame);

            for (auto& detection : detections) {
                detection.depth = getDepth(detection, depthMap);
                detection.className = classNames[detection.classId];
            }

            trackerManager.matchTrackers(detections);
        }
        else {
            trackerManager.updateTrackers();
        }

        // Draw trackers
        trackerManager.drawTrackers(frame);

        // Show/save results
        if (config.output == UserConfig::SHOW || config.output == UserConfig::BOTH) {
            imshow("Tracking", frame);
            cv::waitKey(1);
        }
        if (config.output == UserConfig::SAVE) {
            videoWriter.write(frame);
        }
        if (config.output == UserConfig::RESULTS) {
            trackerManager.outputTrackers(outputFile, frameIdx);
        }
        frameIdx++;
    }

    cout << "Finished processing dataset: " << animalType << "_" << datasetIdx + 1 << endl;
    outputFile.close();
    videoWriter.release();
    listFramesFile.close();
}



//
//int main(void) {
//
//    string filePath = "../img/horse_7/images.txt";
//    string groundTruthPath = "../img/zebra_1/zebra_1_gt.txt";
//
//    // Set up and run user interface
//    UserConfig config;
//    UserInterface ui;
//    ui.run(config);
//
//    // Read out config
//    cout << "Tracker Type: " << config.tracker << endl;
//    cout << "Association Method: " << config.association << endl;
//
//    // Set up YOLO detector
//    const string labelsPath = "../models/coco.names";
//    const string modelPath = "../models/yolo11s-seg.onnx";
//    YOLOv11SegDetector detector(modelPath, labelsPath, true);
//
//    // Set up depth detector
//    const string depthModelPath = "../models/depth_anything_v2_vits.onnx";
//    DepthAnything depthEstimator(depthModelPath, false);
//    cv::Mat depthMap;
//    cv::Mat depthColor;
//
//    // Set up image frames
//    ifstream listFramesFile(filePath);
//    string frameName;
//    Mat frame;
//    int frameIdx = 0;
//
//    // Set up tracker manager
//    vector<string> classNames = ::utils::getClassNames(labelsPath);
//    TrackerManager trackerManager(config, classNames, frame);
//
//    // Run tests
//    if (config.testing == UserConfig::TestingType::MOT_ACCURACY ||
//        config.testing == UserConfig::TestingType::MOT_SWEEP) {
//        auto start = std::chrono::high_resolution_clock::now();
//        TestingHandler testingHandler(config);
//        // Run MOT accuracy tests
//        testingHandler.runTests(groundTruthPath, filePath);
//        auto end = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> elapsed = end - start;
//        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
//        return 0;
//    }
//
//    // Set up video writer
//    string firstFrameName;
//    getline(listFramesFile, firstFrameName);
//    Mat firstFrame = imread(firstFrameName, IMREAD_COLOR);
//    int frameWidth = firstFrame.cols;
//    int frameHeight = firstFrame.rows;
//    double fps = 20.0;
//    string outputVideoPath = "../output/tracking_output.avi";
//    VideoWriter videoWriter;
//    videoWriter.open(outputVideoPath, VideoWriter::fourcc('M','J','P','G'),
//                     fps, Size(frameWidth, frameHeight));
//    listFramesFile.close();
//    listFramesFile.open(filePath);
//
//    // Set up results file
//    std::ofstream outputFile("../results/tracker_results.txt");
//    if (!outputFile.is_open()) {
//        std::cerr << "Failed to open output file: " << "../results/tracker_results.txt" << std::endl;
//        return 1;
//    }
//
//    // Iterate image frames
//    while (getline(listFramesFile, frameName)) {
//        frame = imread(frameName, cv::IMREAD_COLOR);
//        if (frame.empty()) continue;
//        // Redetect every 8 frames
//        if (frameIdx % 8 == 0) {
//            depthMap = depthEstimator.predict(frame.clone());
//            auto detections = detector.segment(frame);
//            // Remove low confidence detections
//
//            for (auto it = detections.begin(); it != detections.end(); ) {
//                if (it->conf <= 0.3) it = detections.erase(it);
//                else ++it;
//            }
//
//            trackerManager.updateTrackersWithDetections(frame, detections, depthMap);
//        }
//        else {
//            trackerManager.updateTrackers(frame);
//        }
//
//        // Visualisation settings
//        if (true) { trackerManager.drawTrackers(frame); }
//        else {
//            // TODO - Set up depth Map visualisation
//            cv::Mat depthVis;
//            cv::normalize(depthMap, depthVis, 0, 255, cv::NORM_MINMAX, CV_8U);
//            cv::applyColorMap(depthVis, depthColor, cv::COLORMAP_JET);
//            cv::Mat blended;
//            cv::addWeighted(frame, 0.4, depthColor, 0.6, 0.0, blended);
//            trackerManager.drawTrackers(blended);
//        }
//
//        // Output settings
//        if (config.output == UserConfig::SHOW || config.output == UserConfig::BOTH) {
//            imshow("Tracking", frame);
//            cv::waitKey(1);
//        }
//        if (config.output == UserConfig::SAVE) videoWriter.write(frame);
//
//        if (config.output == UserConfig::RESULTS) {
//            trackerManager.outputTrackers(outputFile, frameIdx);
//        }
//        ++frameIdx;
//    }
//
//    for (auto& tracker : trackerManager.trackers) {
//        cout << "Tracker " << tracker.trackerNum << endl;
//    }
//
//    return 0;
//}
