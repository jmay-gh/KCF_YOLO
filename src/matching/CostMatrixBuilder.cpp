#include "matching/CostMatrixBuilder.h"

using namespace trackingUtils;

Matrix<float> CostMatrixBuilder::buildCostMatrix(
        std::vector<TrackedObject>& trackers,
        const std::vector<Segmentation>& detections,
        DistanceFunc distanceFunc,
        cv::Mat& frame,
        UserConfig& config,
        float matchingThreshold) {

    int n = trackers.size();
    int m = detections.size();
    Matrix<float> costMatrix(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {

            // Get spatial cost
            float spatialCost = distanceFunc(trackers[i].bbox, toRect(detections[j]));
            // Normalise if it's euclidean distance
            if (config.distance == UserConfig::DistanceType::EUCLIDEAN) {
//                float maxDist = std::sqrt(frame.cols*frame.cols + frame.rows*frame.rows);
//                spatialCost /= maxDist;
                cv::Rect2f tb = trackers[i].bbox;
                float boxDiag = std::sqrt(tb.width * tb.width + tb.height * tb.height);
                spatialCost /= (boxDiag + 1e-6f);
            }

            float secondaryCost = 0.0f;
            // Get area cost
            if (config.auxType == UserConfig::AuxType::AREA) {
                float areaT = trackers[i].bbox.area();
                float areaD = toRect(detections[j]).area();
                secondaryCost = std::abs(areaT - areaD) / std::max(areaT, areaD);
            }
            else if (config.auxType == UserConfig::AuxType::DEPTH) {
                // Get min/max depth across trackers + detections
                auto [minDepth, maxDepth] = getMinMaxDepthDet(trackers, detections);

                float depthT = trackers[i].depth;
                float depthD = detections[j].depth;

                // Skip invalids
                if (depthT < 0 || depthD < 0 || maxDepth == minDepth) {
                    secondaryCost = 1.0f; // maximum penalty
                } else {
                    float normT = (depthT - minDepth) / (maxDepth - minDepth);
                    float normD = (depthD - minDepth) / (maxDepth - minDepth);
                    secondaryCost = std::abs(normT - normD); // ∈ [0,1]
                }
            }
            else if (config.auxType == UserConfig::AuxType::HOG_FEATURES) {
                // Crop detection region
                cv::Rect detectBox = resizeRect(
                        toRect(detections[j]),
                        trackers[i].bbox.width,
                        trackers[i].bbox.height
                );
                detectBox &= cv::Rect(0, 0, frame.cols, frame.rows);

                if (detectBox.empty()) {
                    secondaryCost = 1.0f;
                    std::cout << "No cost" << std::endl;
                    continue;
                }

                // Extract features for the detection
                cv::Mat patch = frame(detectBox).clone();
                cv::Mat detFeatures = trackers[i].tracker.extractFeaturesFromPatch(patch, config.HOG);

                // Now correlate templates directly
                cv::Mat response = trackers[i].tracker.gaussianCorrelation(
                        trackers[i].tracker.getTemplate(),
                        detFeatures
                );

                double maxVal;
                cv::minMaxLoc(response, nullptr, &maxVal, nullptr, nullptr);

                // Use inverse peak as cost
                secondaryCost = 1.0 - maxVal;
            }
            else if (config.auxType == UserConfig::AuxType::VELOCITY) {
                cv::Point2f detectionCenter = trackingUtils::rectCenter(toRect(detections[j]));
                secondaryCost = trackers[i].motionConsistencyScore(detectionCenter);

                // Normalise to [0,1]
                secondaryCost = std::min(1.0f, secondaryCost);
            }

            // Get weighted cost
            if (config.auxType != UserConfig::AuxType::NONE) {
                spatialCost *= 0.2f;
                secondaryCost *= 0.8f;
            }

            float cost = spatialCost + secondaryCost;

            costMatrix(i, j) = cost;
        }
    }
    return costMatrix;
}

using namespace DistanceCalculator;

Matrix<float> CostMatrixBuilder::buildCMEMDMatrix(
        const std::vector<TrackedObject>& trackers,
        const std::vector<Segmentation>& detections,
        cv::Mat& frame,
        UserConfig& config,
        float matchingThreshold) {

    int n = trackers.size();
    int m = detections.size();
    Matrix<float> costMatrix(trackers.size(), detections.size());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (iou(trackers[i].bbox, toRect(detections[j])) > matchingThreshold) {
                auto signatures = SignatureGenerator::generateEMDSignature(trackers[i],
                                                                         detections[j],
                                                                         frame,
                                                                         config);

                float dist = cv::EMD(signatures.first, signatures.second, cv::DIST_L2,
                                     cv::noArray(),nullptr, cv::noArray());
                costMatrix(i, j) = dist;
            }
            else {
                costMatrix(i, j) = std::numeric_limits<float>::max(); // No match
            }
        }
    }
    return costMatrix;
}


Matrix<float> CostMatrixBuilder::buildFMEMDMatrix(
        const std::vector<TrackedObject>& trackers,
        const std::vector<Segmentation>& detections,
        cv::Mat& frame,
        UserConfig& config,
        float matchingThreshold) {

    std::pair<cv::Mat, cv::Mat> sigs = SignatureGenerator::generateMultiSignature(trackers,
                                                                                  detections,
                                                                                  frame,
                                                                                  config);

    cv::Mat flow;
    cv::EMD(sigs.first, sigs.second, cv::DIST_L2,
                    cv::noArray(),nullptr, flow);

//    int n = trackers.size();
//    int m = detections.size();
//    Matrix<float> costMatrix(n, m);
//
//    // Read into Matrix for hung algo
//    for (int r = 0; r < flowMatrix.rows; r++) {
//        int from = static_cast<int>(flowMatrix.at<float>(r, 0));
//        int to   = static_cast<int>(flowMatrix.at<float>(r, 1));
//        float w  = flowMatrix.at<float>(r, 2);
//
//        if (from < n && to < m) {
//            // Accumulate transported weight for this tracker-detection pair
//            costMatrix(from, to) += w;
//        }
//    }

    std::vector<std::tuple<int, int, float>> flowEntries;
    for (int i = 0; i < flow.rows; ++i) {
        for (int j = 0; j < flow.cols; ++j) {
            float f = flow.at<float>(i, j);
            if (f > 0.0f) {
                flowEntries.emplace_back(i, j, f);
            }
        }
    }

    // Convert flow (maximization) to cost matrix (minimization)
    int rows = flow.rows;
    int cols = flow.cols;
    Matrix<float> costMatrix(rows, cols);

    float maxFlow = 0.0f;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            maxFlow = std::max(maxFlow, flow.at<float>(i, j));

    // Fill cost matrix with (maxFlow - actualFlow) to minimize cost
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            costMatrix(i, j) = maxFlow - flow.at<float>(i, j);

    return costMatrix;
}

std::pair<float,float> CostMatrixBuilder::getMinMaxDepthDet(
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