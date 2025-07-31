#pragma once

#include <opencv2/opencv.hpp>
#include "tracking/TrackedObject.h"


class SignatureGenerator {
public:

    static std::pair<cv::Mat, cv::Mat> generateEMDSignature(const TrackedObject& tracker,
                                              const Segmentation& detection,
                                              cv::Mat& frame,
                                              UserConfig& config);

    static cv::Mat computeHOGSignature(const cv::Mat& patch, float weight);

    static cv::Mat computeSpatialSignature1(const cv::Rect& bbox, float depth, float weight);
    static cv::Mat computeSpatialSignature2(const cv::Rect& bbox, float depth, float weight);
    static cv::Mat computeSpatialSignature3(const cv::Rect& bbox, float depth, float weight);
    static cv::Mat computeSpatialSignature4(const cv::Rect& bbox, float depth, float weight);

    static float compareDescriptors(const std::vector<float>& desc1, const std::vector<float>& desc2);
    static float compareSpatialSignatures(const std::vector<float>& sig1, const std::vector<float>& sig2);
};