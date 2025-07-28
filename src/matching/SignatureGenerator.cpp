#include "../include/matching/SignatureGenerator.h"

std::pair<cv::Mat, cv::Mat> SignatureGenerator::generateEMDSignature(const TrackedObject& tracker,
                                                 const Segmentation& detection,
                                                 cv::Mat& frame,
                                                 UserConfig& config) {
    using namespace trackingUtils;

    float trackerWeight;
    float detectionWeight;

    if (config.emdWeight == UserConfig::EMDWeight::UNIFORM) {
        trackerWeight = 1.0f; detectionWeight = 1.0f;
    }
    else if (config.emdWeight == UserConfig::EMDWeight::CONFIDENCE) {
        trackerWeight = tracker.conf; detectionWeight = detection.conf;
    }

    const cv::Rect trackerBox = tracker.bbox;
    const cv::Rect detectionBox = toSafeBox(toRect(detection), frame);

    cv::Mat sig1, sig2;
    switch (config.emdSignature) {
        case UserConfig::EMDSignature::DISTANCE: {
            sig1 = computeSpatialSignature1(trackerBox, tracker.depth, trackerWeight);
            sig2 = computeSpatialSignature1(detectionBox, detection.depth, detectionWeight);
        }
        case UserConfig::EMDSignature::AREA: {
            sig1 = computeSpatialSignature3(trackerBox, tracker.depth, trackerWeight);
            sig2 = computeSpatialSignature3(detectionBox, detection.depth, detectionWeight);
        }
        case UserConfig::EMDSignature::Z_DIST_AREA: {
            sig1 = computeSpatialSignature4(trackerBox, tracker.depth, trackerWeight);
            sig2 = computeSpatialSignature4(detectionBox, detection.depth, detectionWeight);
        }
        case UserConfig::EMDSignature::AVERAGE_HOG: {
            sig1 = computeHOGSignature(frame(trackerBox), trackerWeight);
            sig2 = computeHOGSignature(frame(detectionBox), detectionWeight);
            break;
        }
        default:
            throw std::runtime_error("Unsupported EMD mode in config");
    }
    return {sig1, sig2};
}


// HOG feature-based signature
cv::Mat SignatureGenerator::computeHOGSignature(const cv::Mat& patch, float weight) {
    cv::HOGDescriptor hog(
            cv::Size(64, 128),      // winSize
            cv::Size(16, 16),       // blockSize
            cv::Size(8, 8),     // blockStride
            cv::Size(8, 8),         // cellSize
            9                                       // nbins
    );
    // Resize patch to match HOG input size
    cv::Mat resized;
    cv::resize(patch, resized, hog.winSize);
    std::vector<float> hogFeatures;
    hog.compute(resized, hogFeatures);
    // Create sig and add weight
    cv::Mat signature(1, static_cast<int>(hogFeatures.size() + 1), CV_32F);
    signature.at<float>(0, 0) = weight;
    // Copy HOG features in
    std::memcpy(signature.ptr<float>() + 1, hogFeatures.data(), hogFeatures.size() * sizeof(float));

    if (hogFeatures.empty()) {
        throw std::runtime_error("HOG features are empty — check patch input.");
    }

    return signature;
}


// COMPUTING ALL HOG SIGNATURES
cv::Mat SignatureGenerator::computeSpatialSignature1(const cv::Rect& bbox, float depth, float weight) {
    // (x, y, z, area)
    float x = bbox.x + bbox.width * 0.5f;
    float y = bbox.y + bbox.height * 0.5f;
    cv::Mat sig = (cv::Mat_<float>(1, 3) << weight, x, y);
    return sig;
}

cv::Mat SignatureGenerator::computeSpatialSignature2(const cv::Rect& bbox, float depth, float weight) {
    // (x, y, z, area)
    float x = bbox.x + bbox.width * 0.5f;
    float y = bbox.y + bbox.height * 0.5f;
    float z = depth;
    cv::Mat sig = (cv::Mat_<float>(1, 4) << weight, x, y, z);
    return sig;
}

cv::Mat SignatureGenerator::computeSpatialSignature3(const cv::Rect& bbox, float depth, float weight) {
    // (x, y, z, area)
    float x = bbox.x + bbox.width * 0.5f;
    float y = bbox.y + bbox.height * 0.5f;
    float area = static_cast<float>(bbox.area());
    cv::Mat sig = (cv::Mat_<float>(1, 4) << weight, x, y, area);
    return sig;
}

cv::Mat SignatureGenerator::computeSpatialSignature4(const cv::Rect& bbox, float depth, float weight) {
    // (x, y, z, area)
    float x = bbox.x + bbox.width * 0.5f;
    float y = bbox.y + bbox.height * 0.5f;
    float z = depth;
    float area = static_cast<float>(bbox.area());
    cv::Mat sig = (cv::Mat_<float>(1, 5) << weight, x, y, z, area);
    return sig;
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

