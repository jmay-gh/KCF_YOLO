#include "matching/CostMatrixBuilder.h"

using namespace trackingUtils;
using namespace DistanceCalculator;

Matrix<float> CostMatrixBuilder::buildCostMatrix(
        const std::vector<TrackedObject>& trackers,
        const std::vector<Segmentation>& detections,
        DistanceFunc distanceFunc) {

    int n = trackers.size();
    int m = detections.size();
    Matrix<float> costMatrix(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            costMatrix(i, j) = distanceFunc(trackers[i].bbox, toRect(detections[j]));
        }
    }
    return costMatrix;
}

Matrix<float> CostMatrixBuilder::buildFlowMatrix(
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