#pragma once

#include "TrackedObject.h"
#include "../hungarian_algo/munkres.h"
#include "../seg/YOLO11Seg.hpp"
#include "../matching/MatchingManager.h"

using namespace cv;
using namespace std;

class TrackerManager {

    public:
        TrackerManager(const TrackerConfig& config, const vector<string>& classNames, Mat& currentFrame);

        vector<TrackedObject> trackers;
        Mat& currentFrame;
        MatchingManager matchingManager;

        void updateTrackers(const Mat& frame);
        void updateTrackersWithDetections(const Mat& frame, const vector<Segmentation>& detections, Mat depthMap);
        void drawTrackers(Mat& frame);
        pair<float, float> getMinMaxDepth();

    private:
        int trackerCounter = 0;
        TrackerConfig config;
        vector<string> classNames;

        MatchingManager::MatchResult matchDetections(const vector<Segmentation>& detections);
        Matrix<float> solveCostMatrix(vector<Segmentation> detections);
        vector<int> findUnmatchedDetections(const vector<pair<int, int>>& matches, int totalDetections);
        double getDepth(Segmentation det, Mat depthMap);

};
