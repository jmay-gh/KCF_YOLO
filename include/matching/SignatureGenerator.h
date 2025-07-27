#pragma once

#include <opencv2/opencv.hpp>

class SignatureGenerator {
public:
    static std::vector<float> computeHOGSignature(const cv::Mat& patch, float weight);

    static std::vector<float> computeSpatialSignature(const cv::Rect& bbox, float depth, float weight);
    static float compareDescriptors(const std::vector<float>& desc1, const std::vector<float>& desc2);
    static float compareSpatialSignatures(const std::vector<float>& sig1, const std::vector<float>& sig2);
};