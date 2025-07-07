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
                           const TrackerConfig& config,
                           cv::Mat& frame,
                           MatchingManager& mgr) {
        int positiveMatches = 0;
        int negativeMatches = 0;
        //Create tracked objects from prior ground truths
        std::vector<TrackedObject> trackedObjects;
        for (const auto& gt : prevFrameGT) {
            int w = std::max(1, gt.x + gt.width);
            int h = std::max(1, gt.y + gt.height);
            cv::Mat fakeMatrix(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
            TrackedObject trackedObject(config,
                                        cv::Rect(gt.x, gt.y, gt.width, gt.height),
                                        "",
                                        fakeMatrix,
                                        gt.objectId,
                                        0.0);
            trackedObjects.push_back(trackedObject);
        }

        // Create detections from current ground truths
        std::vector<Segmentation> detectedObjects;
        for (const auto& gt : currFrameGT) {
            Segmentation detectedObject;
            detectedObject.classId = gt.objectId;
            detectedObject.box.x = gt.x + rand() % int(50) - 50 / 2;
            detectedObject.box.y = gt.y + rand() % int(50) - 50 / 2;
            detectedObject.box.width = gt.width;
            detectedObject.box.height = gt.height;
            detectedObjects.push_back(detectedObject);
        }

        MatchingManager::MatchResult matchResult;
        if (config.association == TrackerConfig::AssociationMethod::NEAREST_NEIGHBOUR) {
            matchResult = mgr.matchNN(trackedObjects, detectedObjects);
        }
        else if (config.association == TrackerConfig::AssociationMethod::HUNGARIAN_ALGORITHM) {
            matchResult = mgr.matchHungarian(trackedObjects, detectedObjects);
        }
        else if (config.association == TrackerConfig::AssociationMethod::GROUND_MOVERS_DISTANCE) {
            matchResult = mgr.matchEMD(trackedObjects, detectedObjects, frame);
        }

        for (const auto& match : matchResult.matches) {
            if (match.first == match.second) positiveMatches++;
            else negativeMatches++;
        }
        return std::make_pair(positiveMatches, negativeMatches);
    }

    inline void calculateMOTAccuracy(const std::map<int,
                                     std::vector<GroundTruth>>& groundTruths,
                                     const TrackerConfig& config,
                                     cv::Mat& frame) {
        
        int totalGroundTruths = 0;
        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;

        MatchingManager mgr(config);
        // Collect the matching results
        for (int i = 1; i < groundTruths.size() - 1; i++) {
            if (groundTruths.count(i) && groundTruths.count(i + 1)) {
                auto matches = matchGroundTruths(
                        groundTruths.at(i),
                        groundTruths.at(i + 1),
                        config,
                        frame,
                        mgr
                );
                truePositives += matches.first;
                falseNegatives += matches.second;
                falsePositives += (groundTruths.at(i + 1).size() - matches.first);
            }

//            std::pair<int, int> matches = matchGroundTruths(groundTruths.at(i), groundTruths.at(i+1), config);
//            truePositives += matches.first;
//            falseNegatives += matches.second;
//            falsePositives += (groundTruths.at(i+1).size() - matches.first);
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

}





