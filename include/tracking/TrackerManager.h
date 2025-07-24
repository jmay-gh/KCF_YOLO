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
        float trackConfThreshold;
        float detectConfThreshold;
        int unmatchedThreshold;

        Mat depthMap;

        void updateTrackers(const Mat& frame);
        void updateTrackersWithDetections(const Mat& frame, vector<Segmentation>& detections);
        void drawTrackers(Mat& frame);
        void outputTrackers(std::ofstream& out, int frameIdx);
        pair<float, float> getMinMaxDepth();

    private:
        int trackerCounter = 0;
        TrackerConfig config;
        vector<string> classNames;

        MatchingManager::MatchResult matchDetections(const vector<Segmentation>& detections);

        void matchOccluded(MatchingManager::MatchResult& matchResult,
                           const vector<Segmentation>& detections,
                           Mat depthMap);

        Matrix<float> solveCostMatrix(vector<Segmentation> detections);
        vector<int> findUnmatchedDetections(const vector<pair<int, int>>& matches, int totalDetections);
//        double getDepth(Segmentation det, Mat depthMap);

};
