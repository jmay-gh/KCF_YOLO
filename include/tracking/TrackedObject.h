#pragma once

#include "tracking/KCFTracking/kcftracker.hpp"
#include "userSetup/UserConfig.h"
#include "seg/YOLO11Seg.hpp"
#include "trackingUtils.h"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>

using namespace trackingUtils;

class TrackedObject {

public:
    // Constructor
    TrackedObject(UserConfig& config, Segmentation& detection, cv::Mat& frame, int trackerNum);

    KCFTracker tracker;

    // Tracker id
    int trackerId;
    std::string className;
    cv::Scalar color;

    // Tracker params
    float conf;
    float depth;
    cv::Rect bbox;

    // Tracker states
    bool isMatched;
    bool isOccluded;

    // Tracker states for removal
    int consecutiveLoses = 0;
    int consecutiveMisses = 0;

    // Methods
    bool checkMatched();
    bool checkOccluded();
    void addMiss();
    bool checkMisses();
    bool checkLoss();

    void matchTracker(Segmentation& detection, cv::Mat& frame);
    void updateTracker(cv::Mat& frame);
    void drawTracker(cv::Mat& frame, float minDepth, float maxDepth);
};
