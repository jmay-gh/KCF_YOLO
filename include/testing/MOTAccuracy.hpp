#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "tracking/TrackedObject.h"
#include "seg/YOLO11Seg.hpp"
#include "tracking/KCFTracking/kcftracker.hpp"
#include "tracking/TrackerConfig.h"
#include "matching/MatchingManager.h"

using namespace std;
using namespace cv;

namespace testing {

    struct GroundTruth {
        int frameIdx;
        int objectId;
        int x;
        int y;
        int width;
        int height;
        int confidence;
        int classId;
        int visibility;
    };

    inline void parseGroundTruth(const std::string& filePath, std::map<int, std::vector<GroundTruth>>& groundTruths) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Error opening ground truth file: " << filePath << std::endl;
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream iss(line);
            GroundTruth gt;
            if (!(iss >> gt.frameIdx >> gt.objectId >> gt.x >> gt.y >> gt.width >> gt.height >> gt.confidence >> gt.classId >> gt.visibility)) {
                std::cerr << "Error parsing line: " << line << std::endl;
                continue;
            }
            groundTruths[gt.frameIdx].push_back(gt);
        }
        file.close();
    }


    inline pair<int, int> matchGroundTruths(const std::vector<GroundTruth>& prevFrameGT,
                                            const std::vector<GroundTruth>& currFrameGT,
                                            Mat& prevFrame,
                                            Mat& currFrame,
                                            const TrackerConfig& config,
                                            MatchingManager& mgr) {
        int positiveMatches = 0;
        int negativeMatches = 0;
        //Create tracked objects from prior ground truths
        std::vector<TrackedObject> trackedObjects;
        for (const auto& gt : prevFrameGT) {
            TrackedObject trackedObject(config,
                                        Rect(gt.x, gt.y, gt.width, gt.height),
                                        "",
                                        prevFrame,
                                        gt.objectId,
                                        0.0);
            trackedObjects.push_back(trackedObject);
        }

        // Create detections from current ground truths
        std::vector<Segmentation> detectedObjects;
        for (const auto& gt : currFrameGT) {
            Segmentation detectedObject;
            detectedObject.classId = gt.objectId;
            detectedObject.box = trackingUtils::toBoundingBox(Rect(gt.x, gt.y, gt.width, gt.height));
            detectedObject.box.x = gt.x + rand() % int(40) - 40 / 2;
            detectedObject.box.y = gt.y + rand() % int(40) - 40 / 2;

            std::default_random_engine generator;
            std::normal_distribution<float> scale_dist(0.0f, 0.1f);  // mean = 0, std dev = 10% of size
            float scale_noise_w = scale_dist(generator);
            float scale_noise_h = scale_dist(generator);
            int noisy_width = std::max(1, int(gt.width * (1.0f + scale_noise_w)));
            int noisy_height = std::max(1, int(gt.height * (1.0f + scale_noise_h)));
            detectedObject.box.width = noisy_width;
            detectedObject.box.height = noisy_height;

            detectedObjects.push_back(detectedObject);
        }

        mgr.currentFrame = currFrame;

        MatchingManager::MatchResult matchResult;
        if (config.association == TrackerConfig::AssociationMethod::NEAREST_NEIGHBOUR) {
            matchResult = mgr.matchNN(trackedObjects, detectedObjects);
        }
        else if (config.association == TrackerConfig::AssociationMethod::HUNGARIAN_ALGORITHM) {
            matchResult = mgr.matchHungarian(trackedObjects, detectedObjects);
        }
        else if (config.association == TrackerConfig::AssociationMethod::GROUND_MOVERS_DISTANCE) {
//            matchResult = mgr.matchEMD(trackedObjects, detectedObjects, currFrame, );
        }

        for (const auto& match : matchResult.matches) {
            int prevObjectId = trackedObjects[match.first].trackerNum;
            int currObjectId = detectedObjects[match.second].classId;
            if (prevObjectId == currObjectId) positiveMatches++;
            else {
                negativeMatches++;
//                std::srand(std::time(0));  // Seed the random number generator once
//                int colour = std::rand() % 256; // Gives a value between 0 and 255
//                rectangle(currFrame, trackedObjects[match.first].bbox, Scalar(0, 0, colour), 2);
//                rectangle(currFrame, trackingUtils::toRect(detectedObjects[match.second]), Scalar(0, 0, colour), 2);
//
//                Point trackPoint(trackedObjects[match.first].bbox.x,
//                                  trackedObjects[match.first].bbox.y - 20);
//                putText(currFrame, std::to_string(match.first), trackPoint,
//                        cv::FONT_HERSHEY_SIMPLEX, 0.7, colour, 2);
//
//                Point detectPoint(trackingUtils::toRect(detectedObjects[match.second]).x,
//                                  trackingUtils::toRect(detectedObjects[match.second]).y +
//                                  trackingUtils::toRect(detectedObjects[match.second]).height + 20);
//
//                putText(currFrame, std::to_string(match.second), detectPoint,
//                        cv::FONT_HERSHEY_SIMPLEX, 0.7, colour, 2);
//
//                cout << "Ground Truth Mismatch: Tracker ID " << match.first
//                     << " with Detection ID " << match.second << " in frame: " << prevFrameGT[0].frameIdx << endl;
//                imshow("Ground Truth Mismatch", currFrame);
//                cv::waitKey(0);
            }
        }
        return std::make_pair(positiveMatches, negativeMatches);
    }


    inline void calculateMOTAccuracy(const map<int, vector<GroundTruth>>& groundTruths,
                                     const string& framesFilePath,
                                     const TrackerConfig& config) {

        int totalGroundTruths = 0;
        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;

        MatchingManager mgr(config);
        // Collect the matching results
        std::ifstream framesFile(framesFilePath);
        std::string prevFramePath, currFramePath;
        if (std::getline(framesFile, prevFramePath)) {
            for (int i = 1; i < groundTruths.size() - 1 && getline(framesFile, currFramePath); ++i) {
                cv::Mat prevFrame = cv::imread(prevFramePath);
                cv::Mat currFrame = cv::imread(currFramePath);

//                if (prevFrame.empty()) {
//                    std::cerr << "Failed to load frame: " << prevFramePath << std::endl;
//                }

                if (groundTruths.count(i) && groundTruths.count(i + 1)) {
                    auto matches = matchGroundTruths(
                            groundTruths.at(i),
                            groundTruths.at(i + 1),
                            prevFrame,
                            currFrame,
                            config,
                            mgr
                    );
                    truePositives += matches.first;
                    falseNegatives += matches.second;
                    falsePositives += (groundTruths.at(i + 1).size() - matches.first);
                }
                prevFramePath = currFramePath;
            }
        }

        // Get size of ground truths
        for (int i = 1; i < groundTruths.size() - 1; i++) {
            totalGroundTruths += groundTruths.at(i).size();
        }

        float precision = static_cast<float>(truePositives) / (truePositives + falsePositives);
        float recall = static_cast<float>(truePositives) / (truePositives + falseNegatives);
        float f1Score = 2 * (precision * recall) / (precision + recall);

        std::cout << "Total Ground Truths: " << totalGroundTruths << std::endl;
        std::cout << "True Positives: " << truePositives << std::endl;
        std::cout << "False Positives: " << falsePositives << std::endl;
        std::cout << "False Negatives: " << falseNegatives << std::endl;
        std::cout << "Precision: " << precision << std::endl;
        std::cout << "Recall: " << recall << std::endl;
        std::cout << "F1 Score: " << f1Score << std::endl;
    }


    inline void calculateMOTAccuracyRange(const map<int, vector<GroundTruth>>& groundTruths,
                                          const string& framesFilePath,
                                          const TrackerConfig& config) {

        vector<float> iouThresholds = {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                                       0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0};


        std::ofstream outputFile("../results/iou_sweep.txt");
        if (!outputFile.is_open()) {
            std::cerr << "Failed to open output file: " << "../results/iou_sweep.txt" << std::endl;
            return;
        }

        for (auto iou: iouThresholds) {
            int totalGroundTruths = 0;
            int truePositives = 0;
            int falsePositives = 0;
            int falseNegatives = 0;

            MatchingManager mgr(config);
            mgr.iouThreshold = iou;
            // Collect the matching results
            std::ifstream framesFile(framesFilePath);
            std::string prevFramePath, currFramePath;
            if (std::getline(framesFile, prevFramePath)) {
                for (int i = 1; i < groundTruths.size() - 1 && getline(framesFile, currFramePath); ++i) {
                    cv::Mat prevFrame = cv::imread(prevFramePath);
                    cv::Mat currFrame = cv::imread(currFramePath);

                    if (groundTruths.count(i) && groundTruths.count(i + 1)) {
                        auto matches = matchGroundTruths(
                                groundTruths.at(i),
                                groundTruths.at(i + 1),
                                prevFrame,
                                currFrame,
                                config,
                                mgr
                        );
                        truePositives += matches.first;
                        falseNegatives += matches.second;
                        falsePositives += (groundTruths.at(i + 1).size() - matches.first);
                    }
                    prevFramePath = currFramePath;
                }
            }

            // Get size of ground truths
            for (int i = 1; i < groundTruths.size() - 1; i++) {
                totalGroundTruths += groundTruths.at(i).size();
            }
            float precision = static_cast<float>(truePositives) / (truePositives + falsePositives);
            float recall = static_cast<float>(truePositives) / (truePositives + falseNegatives);
            float f1Score = 2 * (precision * recall) / (precision + recall);

            std::cout << "IoU: " << std::fixed << std::setprecision(2) << iou << std::endl;
            std::cout << "Total Ground Truths: " << totalGroundTruths << std::endl;
            std::cout << "True Positives: " << truePositives << std::endl;
            std::cout << "False Positives: " << falsePositives << std::endl;
            std::cout << "False Negatives: " << falseNegatives << std::endl;
            std::cout << "Precision: " << precision << std::endl;
            std::cout << "Recall: " << recall << std::endl;
            std::cout << "F1 Score: " << f1Score << std::endl;

            // File Output
            outputFile << std::fixed << std::setprecision(2)
                       << iou << ","
                       << totalGroundTruths << ","
                       << truePositives << ","
                       << falsePositives << ","
                       << falseNegatives << ","
                       << precision << ","
                       << recall << ","
                       << f1Score << "\n";
        }
    }

}


