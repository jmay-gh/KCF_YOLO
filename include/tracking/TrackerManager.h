#pragma once

#include "TrackedObject.h"
#include "../hungarian_algo/munkres.h"
#include "../det/YOLO11.hpp"

using namespace cv;
using namespace std;

typedef struct {
    vector<pair<int, int>> matches;
    vector<int> unmatchedTrackers;
    vector<int> unmatchedDetectors;
} MatchResult;

class TrackerManager {

    public:
        TrackerManager(const TrackerConfig& config, const vector<string>& classNames);

        void updateTrackers(const Mat& frame, const vector<Detection>& detections, int frameIdx);
        void drawTrackers(Mat& frame);

    private:
        vector<TrackedObject> trackers;
        int trackerCounter = 0;
        TrackerConfig config;
        vector<string> classNames;

        vector<pair<int, int>> matchDetections(const vector<Detection>& detections);
        Matrix<float> solveCostMatrix(vector<Detection> detections);

        vector<int> findUnmatchedDetections(const vector<pair<int, int>>& matches, int totalDetections);

        Rect toRect(const Detection& det);
        float calculateIOU(const Rect& a, const BoundingBox& b);
};
