#pragma once

#include "matching/AssignmentSolver.h"
#include "matching/CostMatrixBuilder.h"
#include "userSetup/UserConfig.h"
#include "trackingUtils.h"

#include <vector>
#include <iostream>

using namespace std;

class MatchingManager {

public:
    MatchingManager(const UserConfig& config, cv::Mat& frame);

    std::function<float(const cv::Rect&, const cv::Rect&)> distanceFunc;

    UserConfig config;
    cv::Mat& currentFrame;

    float iouThreshold;

    vector<int> trackIds;
    vector<int> detectIds;

    struct MatchResult {
        vector<pair<int, int>> matches;
        set<int> unmatchedTrackers;
        set<int> unmatchedDetections;
    };

    // Matching methods
    MatchResult matchNN(vector<TrackedObject>& trackers, const vector<Segmentation>& detections);
    MatchResult matchHungarian(vector<TrackedObject>& trackers, const vector<Segmentation>& detections);
    MatchResult matchEMD(vector<TrackedObject>& trackers, const vector<Segmentation>& detections);


    // Public distance functions
    double peakResponse(TrackedObject& trackers, KCFTracker& detect, cv::Rect detectBox);

    void normalizeFeatureMap(cv::Mat& feat);

    cv::Mat newSignatureApproach(const vector<cv::Mat>& hogDescriptors, vector<int> hogSizes, vector<int>& objectIds);


    private:

    MatchResult setMatchResult(int trackerSize, int detectionSize);

    // Hungarian
    Matrix<float> computeMatrix(vector<TrackedObject>& trackers,
                                const vector<Segmentation>& detections);



    // EMD

    cv::Mat computeFlow(vector<TrackedObject>& trackers,
            const vector<Segmentation>& detections,
                        cv::Mat& frame,
            std::vector<int>& outTrackLabels,
            std::vector<int>& outDetectionLabels);

    vector<pair<cv::Rect, float>> collectRects(const vector<TrackedObject>& objects);
    vector<pair<cv::Rect, float>> collectRects(const vector<Segmentation>& objects);
    cv::Mat computeSignature(vector<pair<cv::Rect, float>>& boxes, vector<float> weights);

    template <typename T>
    vector<float> collectWeights(const vector<T>& objects);


    cv::Mat resizeMap(const cv::Mat& input);
    cv::Mat fftshift(const cv::Mat& input);
    cv::Mat fourierCropOrPad(const cv::Mat& input, cv::Size targetSize);

    cv::Mat computeHOGSignatureForEMD(const cv::Mat& tmpl);
    cv::Mat spatialPoolHOG(const cv::Mat& tmpl);
    cv::Mat computeHOGSignature(const std::vector<cv::Mat>& hogDescriptors, const std::vector<float>& weights);
    cv::Mat computeAverageHOGSignature(const std::vector<cv::Mat>& hogDescriptors, const std::vector<float>& weights);

};