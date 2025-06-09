#pragma once

#include <opencv2/core/core.hpp>
#include "kcftracker.hpp"
#include "TrackerConfig.h"

class TrackedObject {

public:
    // Constructor
    TrackedObject(TrackerConfig config, cv::Rect bbox, std::string label, cv::Mat frame, int trackerNum);

    // Fields
    KCFTracker tracker;
    cv::Rect bbox;
    std::string label;
    int trackerNum;

    int consecutiveFailures;
    bool matchedDetector;
    cv::Scalar color;

    // Methods
    void setMatched();
    void setUnmatched();
    bool checkMatched();
    void setFailure();
    void resetFailures();
    bool checkFailures();
    void matchTracker(cv::Rect box, cv::Mat frame);
    void draw(cv::Mat& frame);
};
