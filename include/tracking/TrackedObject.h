#pragma once

#include <opencv2/core.hpp>
#include "tracking/KCFTracking/kcftracker.hpp"
#include "TrackerConfig.h"
#include <opencv2/tracking.hpp>

#include <cmath>
#include <algorithm>

class TrackedObject {

public:
    // Constructor
    TrackedObject(TrackerConfig config, cv::Rect bbox, std::string label, cv::Mat frame, int trackerNum, double depth);

    KCFTracker tracker;
    cv::Rect bbox;

    std::string label;
    int trackerNum;

    float conf;
    float depth;

    bool matchedDetector;
    bool isOccluded;

    int consecutiveLoss;
    int consecutiveFailures;

    cv::Scalar color;

    // Methods
    void setMatched();
    void setUnmatched();
    bool checkMatched();
    void setFailure();
    void resetFailures();
    bool checkFailures();

    void matchTracker(cv::Rect box, double depth, const cv::Mat& frame);
    void updateTracker(const cv::Mat& frame);
    void draw(cv::Mat& frame, float minDepth, float maxDepth);
};
