#pragma once

#include <vector>
#include <iostream>

#include "../tracking/TrackedObject.h"
#include "../seg/YOLO11Seg.hpp"
#include "tracking/KCFTracking/kcftracker.hpp"
#include "../trackingUtils.h"
#include "../hungarian_algo/matrix.h"
#include "../tracking/KCFTracking/fhog.hpp"

#include "../tracking/KCFTracking/ffttools.hpp"
#include "../tracking/KCFTracking/recttools.hpp"

#include "../hungarian_algo/munkres.h"

using namespace std;

class MatchingManager {


    public:

        cv::Mat currentFrame;
        float iouThreshold;

        vector<int> trackIds;
        vector<int> detectIds;

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
        double peakResponse(TrackedObject& trackers, KCFTracker& detect, Rect detectBox);

        void normalizeFeatureMap(cv::Mat& feat);

        Mat newSignatureApproach(const std::vector<cv::Mat>& hogDescriptors, vector<int> hogSizes, vector<int>& objectIds);


        private:

        MatchResult setMatchResult(int trackerSize, int detectionSize);

        // Hungarian
        Matrix<float> computeMatrix(vector<TrackedObject>& trackers,
                                    const vector<Segmentation>& detections);



        // EMD

        Mat computeFlow(vector<TrackedObject>& trackers,
                const vector<Segmentation>& detections,
                Mat& frame,
                std::vector<int>& outTrackLabels,
                std::vector<int>& outDetectionLabels);

        vector<pair<Rect, float>> collectRects(const vector<TrackedObject>& objects);
        vector<pair<Rect, float>> collectRects(const vector<Segmentation>& objects);
        Mat computeSignature(vector<pair<Rect, float>>& boxes, vector<float> weights);

        template <typename T>
        std::vector<float> collectWeights(const vector<T>& objects);

        Mat resizeMap(const Mat& input);
        Mat fftshift(const cv::Mat& input);
        Mat fourierCropOrPad(const cv::Mat& input, cv::Size targetSize);

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