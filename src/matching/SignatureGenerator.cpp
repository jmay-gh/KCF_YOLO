#include "../include/matching/SignatureGenerator.h"

std::pair<cv::Mat, cv::Mat> SignatureGenerator::generateEMDSignature(const TrackedObject& tracker,
                                                 const Segmentation& detection,
                                                 cv::Mat& frame,
                                                 UserConfig& config) {
    using namespace trackingUtils;

    float trackerWeight = 1.0f;
    float detectionWeight = 1.0f;

    if (config.emdWeight == UserConfig::EMDWeight::CONFIDENCE) {
        trackerWeight = tracker.conf; detectionWeight = detection.conf;
    }

    const cv::Rect trackerBox = tracker.bbox;
    const cv::Rect detectionBox = toSafeBox(toRect(detection), frame);

    cv::Mat sig1, sig2;
    switch (config.auxType) {
        case UserConfig::AuxType::NONE: {
            sig1 = computeDistSig(trackerBox, tracker.depth, trackerWeight);
            sig2 = computeDistSig(detectionBox, detection.depth, detectionWeight);
            break;
        }
        case UserConfig::AuxType::AREA: {
            sig1 = computeAreaSig(trackerBox, tracker.depth, trackerWeight);
            sig2 = computeAreaSig(detectionBox, detection.depth, detectionWeight);
            break;
        }
        case UserConfig::AuxType::DEPTH: {
            sig1 = computeDepthSig(trackerBox, tracker.depth, trackerWeight);
            sig2 = computeDepthSig(detectionBox, detection.depth, detectionWeight);
            break;
        }
        case UserConfig::AuxType::HOG_FEATURES: {
            float trackX = tracker.bbox.x + tracker.bbox.width * 0.5f;
            float trackY = tracker.bbox.y + tracker.bbox.height * 0.5f;
            float detX = detectionBox.x + detectionBox.width * 0.5f;
            float detY = detectionBox.y + detectionBox.height * 0.5f;
            sig1 = computeHOGSignature(frame(trackerBox), trackerWeight, trackX, trackY);
            sig2 = computeHOGSignature(frame(detectionBox), detectionWeight, detX, detY);
            break;
        }
        case UserConfig::AuxType::VELOCITY: {
            sig1 = computeVelocitySig(trackerBox, 0.0f, trackerWeight);
            sig2 = computeVelocitySig(detectionBox,
                                      tracker.motionConsistencyScore(rectCenter(detectionBox)),
                                      detectionWeight);
            break;
        }
        default:
            throw std::runtime_error("Unsupported EMD mode in config");
    }
    return {sig1, sig2};
}


cv::Mat SignatureGenerator::computeDistSig(const cv::Rect& bbox, float depth, float weight) {
    // (x, y, z, area)
    float x = bbox.x + bbox.width * 0.5f;
    float y = bbox.y + bbox.height * 0.5f;
    cv::Mat sig = (cv::Mat_<float>(1, 3) << weight, x, y);
    return sig;
}


cv::Mat SignatureGenerator::computeAreaSig(const cv::Rect& bbox, float depth, float weight) {
    float x = bbox.x + bbox.width * 0.5f;
    float y = bbox.y + bbox.height * 0.5f;

    cv::Mat sig = (cv::Mat_<float>(5, 3) <<
            0.2f, x, y,                                 // center
            0.2f, bbox.x, bbox.y,                       // top-left
            0.2f, bbox.x+bbox.width, bbox.y,            // top-right
            0.2f, bbox.x, bbox.y+bbox.height,           // bottom-left
            0.2f, bbox.x+bbox.width, bbox.y+bbox.height // bottom-right
    );
//    cv::Mat sig = (cv::Mat_<float>(1, 4) << weight, x, y, area);

    return sig;
}


cv::Mat SignatureGenerator::computeDepthSig(const cv::Rect& bbox, float depth, float weight) {
    float x = bbox.x + bbox.width * 0.5f;
    float y = bbox.y + bbox.height * 0.5f;
    float z = depth;

    cv::Mat sig = (cv::Mat_<float>(2, 4) <<
            0.5f * weight, x, y, 0.0f,          // spatial cluster (z = dummy)
            0.5f * weight, 0.0f, 0.0f, z        // depth cluster (x,y = dummy)
    );
//    cv::Mat sig = (cv::Mat_<float>(1, 4) << weight, x, y, z);

    return sig;
}


cv::Mat SignatureGenerator::computeMultiHOGSignature(const cv::Mat& patch,
                                                     float weight,
                                                     int gridX,
                                                     int gridY) {
    // HOG descriptor
    cv::HOGDescriptor hog(
            cv::Size(64, 128),  // winSize
            cv::Size(16, 16),   // blockSize
            cv::Size(8, 8),     // blockStride
            cv::Size(8, 8),     // cellSize
            9                   // nbins
    );
    const int hogSize = hog.getDescriptorSize();

    // Each row = [ weight | x | y | HOG... ]
    const int cols = 1 + 2 + hogSize;
    cv::Mat signature(gridX * gridY, cols, CV_32F, cv::Scalar(0.0f));

    int row = 0;
    for (int gy = 0; gy < gridY; ++gy) {
        for (int gx = 0; gx < gridX; ++gx) {
            // Extract sub-patch
            int x0 = gx * patch.cols / gridX;
            int y0 = gy * patch.rows / gridY;
            int w  = (gx == gridX - 1) ? patch.cols - x0 : patch.cols / gridX;
            int h  = (gy == gridY - 1) ? patch.rows - y0 : patch.rows / gridY;
            cv::Rect cellRect(x0, y0, w, h);

            cv::Mat cell = patch(cellRect);
            cv::Mat resized;
            cv::resize(cell, resized, hog.winSize);

            // Compute HOG
            std::vector<float> hogFeatures;
            hog.compute(resized, hogFeatures);

            // Row pointer
            float* ptr = signature.ptr<float>(row);

            // Assign weight (even split among cells)
            ptr[0] = weight / static_cast<float>(gridX * gridY);

            // Normalized sub-patch center coords
            float cx = (x0 + w * 0.5f) / static_cast<float>(patch.cols);
            float cy = (y0 + h * 0.5f) / static_cast<float>(patch.rows);
            ptr[1] = cx;
            ptr[2] = cy;

            // Copy HOG features
            std::memcpy(ptr + 3, hogFeatures.data(), hogSize * sizeof(float));

            ++row;
        }
    }

    return signature;
}


cv::Mat SignatureGenerator::computeVelocitySig(const cv::Rect& bbox, float motionScore, float weight) {
    float x = bbox.x + bbox.width * 0.5f;
    float y = bbox.y + bbox.height * 0.5f;
    float ms = motionScore;

    cv::Mat sig = (cv::Mat_<float>(2, 4) <<
            0.5f * weight, x, y, 0.0f,          // spatial cluster (z = dummy)
            0.5f * weight, 0.0f, 0.0f, ms        // depth cluster (x,y = dummy)
    );
//    cv::Mat sig = (cv::Mat_<float>(1, 4) << weight, x, y, z);

    return sig;
}


// HOG feature-based signature
cv::Mat SignatureGenerator::computeHOGSignature(const cv::Mat& patch, float weight, float x, float y) {
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

    // Normalize HOG features
    cv::Mat signature(1, static_cast<int>(hogFeatures.size() + 3), CV_32F);

    signature.at<float>(0, 0) = weight;
    signature.at<float>(0, 1) = x;
    signature.at<float>(0, 2) = y;

    // Copy HOG features in
    std::memcpy(signature.ptr<float>() + 3, hogFeatures.data(), hogFeatures.size() * sizeof(float));
    if (hogFeatures.empty()) {
        throw std::runtime_error("HOG features are empty — check patch input.");
    }
    return signature;
}


std::pair<cv::Mat, cv::Mat> SignatureGenerator::generateMultiSignature(
        const std::vector<TrackedObject>& trackers,
        const std::vector<Segmentation>& detections,
        cv::Mat& frame,
        UserConfig& config)
{
    cv::Mat trackerSig, detectionSig;

    auto [minDepth, maxDepth] = getMinMaxDepthDet(trackers, detections);

    // --- Trackers ---
    for (const auto& tracker : trackers) {
        cv::Mat sigRow;
        switch (config.auxType) {
            case UserConfig::AuxType::NONE: {
                float x = (tracker.bbox.x + tracker.bbox.width * 0.5f) / static_cast<float>(frame.cols);
                float y = (tracker.bbox.y + tracker.bbox.height * 0.5f) / static_cast<float>(frame.rows);
                sigRow = (cv::Mat_<float>(1, 3) << tracker.conf, x, y);
                break;
            }
            case UserConfig::AuxType::AREA: {
                float x = (tracker.bbox.x + tracker.bbox.width * 0.5f) / static_cast<float>(frame.cols);
                float y = (tracker.bbox.y + tracker.bbox.height * 0.5f) / static_cast<float>(frame.rows);
                float areaNorm = tracker.bbox.area() / static_cast<float>(frame.cols * frame.rows);
                sigRow = (cv::Mat_<float>(1, 4) << tracker.conf, x, y, areaNorm);
                break;
            }
            case UserConfig::AuxType::DEPTH: {
                float x = (tracker.bbox.x + tracker.bbox.width * 0.5f) / static_cast<float>(frame.cols);
                float y = (tracker.bbox.y + tracker.bbox.height * 0.5f) / static_cast<float>(frame.rows);

                float zNorm = 0.5f; // default fallback
                if (tracker.depth >= 0) {
                    zNorm = (tracker.depth - minDepth) / (maxDepth - minDepth); // ∈ [0,1]
                }

                sigRow = (cv::Mat_<float>(1, 4) << tracker.conf, x, y, zNorm);
                break;
            }
            case UserConfig::AuxType::VELOCITY: {
                float x = tracker.bbox.x + tracker.bbox.width * 0.5f;
                float y = tracker.bbox.y + tracker.bbox.height * 0.5f;
                cv::Point2f predictedPos = tracker.predictNextPosition();
                sigRow = (cv::Mat_<float>(1, 5) << tracker.conf, x, y, predictedPos.x, predictedPos.y);
                break;
            }
            case UserConfig::AuxType::HOG_FEATURES: {
                float x = tracker.bbox.x + tracker.bbox.width * 0.5f;
                float y = tracker.bbox.y + tracker.bbox.height * 0.5f;
                sigRow = computeHOGSignature(frame(tracker.bbox), tracker.conf, x, y);
                break;
            }
        }
        trackerSig.push_back(sigRow); // vconcat alternative
    }

    // --- Detections ---
    for (const auto& det : detections) {
        cv::Mat sigRow;
        switch (config.auxType) {
            case UserConfig::AuxType::NONE: {
                float x = (det.box.x + det.box.width * 0.5f) / static_cast<float>(frame.cols);
                float y = (det.box.y + det.box.height * 0.5f) / static_cast<float>(frame.rows);
                sigRow = (cv::Mat_<float>(1, 3) << det.conf, x, y);
                break;
            }
            case UserConfig::AuxType::AREA: {
                float x = (det.box.x + det.box.width * 0.5f) / static_cast<float>(frame.cols);
                float y = (det.box.y + det.box.height * 0.5f) / static_cast<float>(frame.rows);
                float areaNorm = det.box.area() / static_cast<float>(frame.cols * frame.rows);
                sigRow = (cv::Mat_<float>(1, 4) << det.conf, x, y, areaNorm);
                break;
            }
            case UserConfig::AuxType::DEPTH: {
                float x = (det.box.x + det.box.width * 0.5f) / static_cast<float>(frame.cols);
                float y = (det.box.y + det.box.height * 0.5f) / static_cast<float>(frame.rows);

                float zNorm = 0.5f; // default fallback
                if (det.depth >= 0) {
                    zNorm = (det.depth - minDepth) / (maxDepth - minDepth); // ∈ [0,1]
                }

                sigRow = (cv::Mat_<float>(1, 4) << det.conf, x, y, zNorm);
                break;
            }
            case UserConfig::AuxType::VELOCITY: {
                float x = det.box.x + det.box.width * 0.5f;
                float y = det.box.y + det.box.height * 0.5f;
                sigRow = (cv::Mat_<float>(1, 5) << det.conf, x, y, x, y);
                break;
            }
            case UserConfig::AuxType::HOG_FEATURES: {
                float x = det.box.x + det.box.width * 0.5f;
                float y = det.box.y + det.box.height * 0.5f;
                sigRow = computeHOGSignature(frame(toRect(det)), det.conf, x, y);
                break;
            }
        }
        detectionSig.push_back(sigRow);
    }

    return {trackerSig, detectionSig};
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

std::pair<float,float> SignatureGenerator::getMinMaxDepthDet(
        const std::vector<TrackedObject>& trackers,
        const std::vector<Segmentation>& detections)
{
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();

    for (const auto& t : trackers) {
        if (t.depth < 0) continue;
        minVal = std::min(minVal, t.depth);
        maxVal = std::max(maxVal, t.depth);
    }
    for (const auto& d : detections) {
        if (d.depth < 0) continue;
        minVal = std::min(minVal, d.depth);
        maxVal = std::max(maxVal, d.depth);
    }

    if (minVal == std::numeric_limits<float>::max()) {
        return {0.0f, 0.0f};
    }
    return {minVal, maxVal};
}
