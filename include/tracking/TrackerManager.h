#pragma once

#include "matching/MatchingManager.h"
#include "tracking/TrackedObject.h"

using namespace std;

class TrackerManager {
public:
    TrackerManager(const UserConfig& config, cv::Mat& frame);

    // Current trackers
    vector<TrackedObject> trackers;

    // Current frame
    cv::Mat& currentFrame;

    // Matching manager class
    MatchingManager matchingManager;
    using MatchFunction = function<MatchingManager::MatchResult(vector<TrackedObject>&,
                                                                const vector<Segmentation>&)>;
    MatchFunction matchFunc;

    // Threshold configuration parameters
    float trackConfThreshold;
    float detectConfThreshold;
    int lossThreshold;

    // Update trackers with the current frame
    void updateTrackers();

    // Match trackers to detections
    void matchTrackers(vector<Segmentation>& detections);
    void matchOccludedTrackers(MatchingManager::MatchResult& matchResult,
                               vector<Segmentation>& detections);

    // Draw and record trackers
    void drawTrackers(cv::Mat& frame);
    void outputTrackers(std::ofstream& out, int frameIdx);

    pair<float, float> getMinMaxDepth();

    int trackerCounter = 0;
    UserConfig config;
};
