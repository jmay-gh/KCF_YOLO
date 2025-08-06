#pragma once

#include "matching/MatchingManager.h"
#include "tracking/TrackedObject.h"

using namespace std;

class TrackerManager {
public:
    TrackerManager(UserConfig config, cv::Mat& frame);

    // Current trackers
    vector<TrackedObject> trackers;

    // Current frame
    cv::Mat& currentFrame;

    // Matching manager class
    MatchingManager matchingManager;
    using MatchFunction = function<MatchingManager::MatchResult(vector<TrackedObject>&,
                                                                const vector<Segmentation>&,
                                                                float threshold)>;
    MatchFunction matchFunc;

    // Threshold configuration parameters
    float trackConfThreshold = 0.4f;
    int lossThreshold = 9;
    int nomatchThreshold = 9;

    // Relaxed occlusion matching parameters
    float occConfThreshold;
    int occLossThreshold;

    // Matching thresholds
    float baseMatchThreshold;
    float thresholdMultiple;

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
