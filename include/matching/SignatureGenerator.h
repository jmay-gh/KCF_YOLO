#pragma once

#include <opencv2/opencv.hpp>
#include "tracking/TrackedObject.h"


class SignatureGenerator {
public:

    static std::pair<cv::Mat, cv::Mat> generateMultiSignature(
            const std::vector<TrackedObject>& trackers,
            const std::vector<Segmentation>& detections,
            cv::Mat& frame,
            UserConfig& config);

    static std::pair<cv::Mat, cv::Mat> generateEMDSignature(const TrackedObject& tracker,
                                              const Segmentation& detection,
                                              cv::Mat& frame,
                                              UserConfig& config);
    // Signature for Dist
    static cv::Mat computeDistSig(const cv::Rect& bbox, float depth, float weight);

    // Signature for Area
    static cv::Mat computeAreaSig(const cv::Rect& bbox, float depth, float weight);

    // Signature for Depth
    static cv::Mat computeDepthSig(const cv::Rect& bbox, float depth, float weight);

    // Signature for HOG
    static cv::Mat computeHOGSignature(const cv::Mat& patch, float weight, float x, float y);

    // Signature for velocity
    static cv::Mat computeVelocitySig(const cv::Rect& bbox, float motionScore, float weight);

    // Signature for multi-hog (better)
    static cv::Mat computeMultiHOGSignature(const cv::Mat& patch, float weight, int gridX, int gridY);

    static float compareDescriptors(const std::vector<float>& desc1, const std::vector<float>& desc2);
    static float compareSpatialSignatures(const std::vector<float>& sig1, const std::vector<float>& sig2);

    static std::pair<float,float> getMinMaxDepthDet(
            const std::vector<TrackedObject>& trackers,
            const std::vector<Segmentation>& detections);
};