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
    bool isMatched = true;
    bool isOccluded = false;
    bool occRemoval = false;
    bool depthOn = false;

    // Tracker states for removal
    float confThreshold;
    float occConfThreshold;

    int consecutiveLoses = 0;
    int consecutiveMisses = 0;

    // Velocity params
    std::deque<cv::Point2f> positionHistory;
    cv::Mat motionSubspace;   // cached subspace basis
    int hankelRows = 3;       // tunable
    size_t motionWindowSize = 10;

    // Methods
    bool checkMatched();
    bool checkOccluded();

    void addMiss();
    void addLoss();

    bool checkMisses();
    bool checkLoss();

    void matchTracker(Segmentation& detection, cv::Mat& frame);
    void updateTracker(cv::Mat& frame);
    void drawTracker(cv::Mat& frame, float minDepth, float maxDepth);


    void updateMotionHistory();
    cv::Mat buildHankel(const std::vector<float>& seq, int rows) const;
    cv::Mat computeMotionSubspace(const std::deque<cv::Point2f>& history) const;
    float motionConsistencyScore(const cv::Point2f& candidate) const;
    cv::Point2f predictNextPosition() const;

    };
