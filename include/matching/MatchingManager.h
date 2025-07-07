#pragma once

#include <vector>
#include <iostream>

#include "../tracking/TrackedObject.h"
#include "../seg/YOLO11Seg.hpp"
#include "tracking/KCFTracking/kcftracker.hpp"
#include "../trackingUtils.h"
#include "../hungarian_algo/matrix.h"
#include "../hungarian_algo/munkres.h"

using namespace std;

class MatchingManager {

    public:
        struct MatchResult {
            vector<pair<int, int>> matches;
            set<int> unmatchedTrackers;
            set<int> unmatchedDetections;
        };

        MatchingManager(const TrackerConfig& config);
        TrackerConfig config;

        // Public matching interfaces
        MatchResult matchNN(vector<TrackedObject>& trackers,
                                      const vector<Segmentation>& detections);

        MatchResult matchHungarian(vector<TrackedObject>& trackers,
                                   const vector<Segmentation>& detections);

        MatchResult matchEMD(vector<TrackedObject>& trackers,
                             const vector<Segmentation>& detections,
                             Mat& frame);

        // Public distance functions
        double euclidean(const cv::Rect& aRect, const cv::Rect& bRect);
        double iou(const cv::Rect& a, const cv::Rect& b);
        double inverseIou(const cv::Rect& a, const cv::Rect& b);

    private:

        MatchResult setMatchResult(int trackerSize, int detectionSize);

        // Hungarian
        Matrix<float> computeMatrix(vector<TrackedObject>& trackers,
                                    const vector<Segmentation>& detections);

        // EMD

        Mat computeFlow(vector<TrackedObject>& trackers,
                const vector<Segmentation>& detections,
                Mat& frame);

        vector<Rect> collectRects(const vector<TrackedObject>& objects);
        vector<Rect> collectRects(const vector<Segmentation>& objects);
        Mat computeSignature(vector<Rect>& boxes, vector<float> weights);

        template <typename T>
        std::vector<float> collectWeights(const vector<T>& objects);

        cv::Mat computeHOGSignatureForEMD(const cv::Mat& tmpl);
        cv::Mat spatialPoolHOG(const cv::Mat& tmpl);
        cv::Mat computeHOGSignature(const std::vector<cv::Mat>& hogDescriptors, const std::vector<float>& weights);
        cv::Mat computeAverageHOGSignature(const std::vector<cv::Mat>& hogDescriptors, const std::vector<float>& weights);

};

namespace distances {
    double euclidean(const cv::Rect& aRect, const cv::Rect& bRect);
    double iou(const cv::Rect& a, const cv::Rect& b);
    double inverseIou(const cv::Rect& a, const cv::Rect& b);
}