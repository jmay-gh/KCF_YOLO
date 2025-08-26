#pragma once

#include "tracking/TrackerManager.h"
#include "depth/depth_anything.hpp"
#include "userSetup/UserInterface.h"

#include <opencv2/videoio.hpp>
#include <filesystem>
#include <chrono>


using namespace std;
using namespace trackingUtils;
namespace fs = std::filesystem;

void runTrackingOnDataset(const std::string& filePath,
                          const std::string& groundTruthPath,
                          UserConfig& config,
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

            {"../img/chicken_1/images.txt", "../img/chicken_1/chicken_1_gt.txt"},
            {"../img/chicken_3/images.txt", "../img/chicken_3/chicken_3_gt.txt"},
            {"../img/chicken_4/images.txt", "../img/chicken_4/chicken_4_gt.txt"},
            {"../img/chicken_5/images.txt", "../img/chicken_5/chicken_5_gt.txt"},

            {"../img/horse_1/images.txt", "../img/horse_1/horse_1_gt.txt"},
            {"../img/horse_2/images.txt", "../img/horse_2/horse_2_gt.txt"},
            {"../img/horse_4/images.txt", "../img/horse_4/horse_4_gt.txt"},
            {"../img/horse_5/images.txt", "../img/horse_5/horse_5_gt.txt"},
            {"../img/horse_6/images.txt", "../img/horse_6/horse_6_gt.txt"},

            {"../img/deer_1/images.txt", "../img/deer_1/deer_1_gt.txt"},
            {"../img/deer_2/images.txt", "../img/deer_2/deer_2_gt.txt"},
            {"../img/deer_3/images.txt", "../img/deer_3/deer_3_gt.txt"},
            {"../img/deer_4/images.txt", "../img/deer_4/deer_4_gt.txt"},
            {"../img/deer_5/images.txt", "../img/deer_5/deer_5_gt.txt"},
            {"../img/deer_6/images.txt", "../img/deer_6/deer_6_gt.txt"},
            {"../img/deer_7/images.txt", "../img/deer_7/deer_7_gt.txt"},

            {"../img/zebra_1/images.txt", "../img/zebra_1/zebra_1_gt.txt"},
            {"../img/zebra_2/images.txt", "../img/zebra_2/zebra_2_gt.txt"},
            {"../img/zebra_3/images.txt", "../img/zebra_3/zebra_3_gt.txt"},
            {"../img/zebra_4/images.txt", "../img/zebra_4/zebra_4_gt.txt"},
            {"../img/zebra_5/images.txt", "../img/zebra_5/zebra_5_gt.txt"},

            {"../img/pig_1/images.txt", "../img/pig_1/pig_1_gt.txt"},
            {"../img/pig_2/images.txt", "../img/pig_2/pig_2_gt.txt"},
            {"../img/pig_3/images.txt", "../img/pig_3/pig_3_gt.txt"},
            {"../img/pig_4/images.txt", "../img/pig_4/pig_4_gt.txt"},
            {"../img/pig_5/images.txt", "../img/pig_5/pig_5_gt.txt"},

            {"../img/penguin_1/images.txt", "../img/penguin_1/penguin_1_gt.txt"},
            {"../img/penguin_2/images.txt", "../img/penguin_2/penguin_2_gt.txt"},
            {"../img/penguin_3/images.txt", "../img/penguin_3/penguin_3_gt.txt"},
            {"../img/penguin_4/images.txt", "../img/penguin_4/penguin_4_gt.txt"},
            {"../img/penguin_5/images.txt", "../img/penguin_5/penguin_5_gt.txt"},
            {"../img/penguin_6/images.txt", "../img/penguin_6/penguin_6_gt.txt"},
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
                          UserConfig& config,
                          YOLOv11SegDetector& detector,
                          DepthAnything& depthEstimator,
                          const std::vector<std::string>& classNames,
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

        // Get folder name and set file name
        std::filesystem::path p(filePath);
        std::string folderName = p.parent_path().filename().string();
        std::string fileName = folderName + ".txt";

        // Construct output directory and file path
        fs::path relativeOutputPath = "test/data/" + fileName;
        // Create output directory if it doesn't exist
        string outputVideoPath = "../results/" + folderName + ".avi";
        cv::VideoWriter videoWriter(outputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                    fps, cv::Size(frameWidth, frameHeight));

        fs::path resultsPath =
                "../results/speed_test/tracker_results/" +
                relativeOutputPath.string();
        fs::create_directories(fs::path(resultsPath).parent_path());

        ofstream outputFile(resultsPath);
        if (!outputFile.is_open()) {
            cerr << "Could not write to: " << resultsPath << endl;
            return;
        }

        // Tracker manager
        cv::Mat frame, depthMap, depthColor;
        TrackerManager trackerManager(config, frame);

//        trackerManager.occConfThreshold = relaxedRemoval;
//        trackerManager.thresholdMultiple = relaxedMatching;
//        trackerManager.baseMatchThreshold = matchThreshold;

        auto startTime = std::chrono::high_resolution_clock::now();

        int frameIdx = 0;
        while (getline(listFramesFile, frameName)) {
            frame = imread(frameName, cv::IMREAD_COLOR);

            if (frame.empty()) {
                std::cerr << "Could not load frame from path!" << std::endl;
                std::exit(1);
            }

            if (frame.empty()) continue;

            if (frameIdx % 8 == 0) {
                if (config.auxType == UserConfig::AuxType::DEPTH) {
                    depthMap = depthEstimator.predict(frame);
                }

                auto detections = detector.segment(frame);

                for (auto &detection: detections) {
                    if (config.auxType == UserConfig::AuxType::DEPTH) {
                        detection.depth = getDepth(detection, depthMap);
                    }
                    else {
                        detection.depth = 0.0f;
                    }
                    detection.className = classNames[detection.classId];
                }
                trackerManager.matchTrackers(detections);
            } else {
                trackerManager.updateTrackers();
            }

            // Draw trackers
            trackerManager.drawTrackers(frame);

            // Show/save results
            if (config.output == UserConfig::SHOW || config.output == UserConfig::BOTH) {
                cv::Mat smallFrame;
                cv::resize(frame, smallFrame, cv::Size(), 0.5, 0.5);
                imshow("Tracking", frame);
                cv::waitKey(30);
            }
            if (config.output == UserConfig::SAVE) {
                videoWriter.write(frame);
            }
            if (config.output == UserConfig::RESULTS) {
                trackerManager.outputTrackers(outputFile, frameIdx);
            }
            frameIdx++;
        }

        cout << "Finished processing dataset: " << folderName << endl;

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;

        double fpsMeasured = frameIdx / elapsed.count();

        cout << "Finished processing dataset: " << folderName
             << " | FPS: " << fpsMeasured << endl;

        // Append FPS to a log file
        std::ofstream fpsLog("../results/speed_test/fps_log.txt", std::ios::app);
        if (fpsLog.is_open()) {
            fpsLog << folderName << " " << fpsMeasured << std::endl;
            fpsLog.close();
        }

        outputFile.close();
        videoWriter.release();
        listFramesFile.close();
}

