#include "../include/matching/SignatureGenerator.h"

// HOG feature-based signature
std::vector<float> SignatureGenerator::computeHOGSignature(const cv::Mat& patch, float weight) {
    cv::HOGDescriptor hog(
            cv::Size(64, 128),      // winSize
            cv::Size(16, 16),       // blockSize
            cv::Size(8, 8),     // blockStride
            cv::Size(8, 8),         // cellSize
            9                                       // nbins
    );

    // Compute the HOG features
    std::vector<float> hogFeatures;
    cv::Mat resized;
    cv::resize(patch, resized, hog.winSize);
    hog.compute(resized, hogFeatures);

    // Weight the HOG features
    std::vector<float> descriptors = {weight};
    descriptors.insert(descriptors.end(), hogFeatures.begin(), hogFeatures.end());
    return descriptors;
}

std::vector<float> SignatureGenerator::computeSpatialSignature(const cv::Rect& bbox, float depth, float weight) {
    // (x, y, z, area)
    float x = bbox.x + bbox.width * 0.5f;
    float y = bbox.y + bbox.height * 0.5f;
    float z = depth;
    float area = static_cast<float>(bbox.area());
    return {weight, x, y, z, area};
}

float SignatureGenerator::compareDescriptors(const std::vector<float>& desc1, const std::vector<float>& desc2) {
    // Cosine distance
    float dot = 0.f, norm1 = 0.f, norm2 = 0.f;
    for (size_t i = 0; i < desc1.size(); ++i) {
        dot += desc1[i] * desc2[i];
        norm1 += desc1[i] * desc1[i];
        norm2 += desc2[i] * desc2[i];
    }
    return 1.0f - dot / (std::sqrt(norm1) * std::sqrt(norm2) + 1e-6f);
}

float SignatureGenerator::compareSpatialSignatures(const std::vector<float>& sig1, const std::vector<float>& sig2) {
    // Simple normalized Euclidean distance
    float dist = 0.0f;
    for (size_t i = 0; i < sig1.size(); ++i) {
        float d = sig1[i] - sig2[i];
        dist += d * d;
    }
    return std::sqrt(dist);
}

