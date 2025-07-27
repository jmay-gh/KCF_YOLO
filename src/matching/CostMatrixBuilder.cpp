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
            if (ios(trackers[i].bbox, toRect(detections[j])) > 0.5f) {
                costMatrix(i, j) = distanceFunc(trackers[i].bbox, toRect(detections[j]));
            } else {
                costMatrix(i, j) = std::numeric_limits<float>::max(); // No match
            }
        }
    }
    return costMatrix;
}

Matrix<float> CostMatrixBuilder::buildFlowMatrix(
        const std::vector<TrackedObject>& trackers,
        const std::vector<Segmentation>& detections,
        cv::Mat& frame) {

    int n = trackers.size();
    int m = detections.size();
    Matrix<float> costMatrix(trackers.size(), detections.size());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (ios(trackers[i].bbox, toRect(detections[j])) > 0.5f) {

//                auto sig1 = SignatureGenerator::computeHOGSignature(
//                        frame(trackers[i].bbox), trackers[i].conf);
//                auto sig2 = SignatureGenerator::computeHOGSignature(
//                        frame(toSafeBox(toRect(detections[j]), frame)), detections[j].conf);;

                auto sig1 = SignatureGenerator::computeSpatialSignature(
                        trackers[i].bbox, trackers[i].depth, trackers[i].conf);
                auto sig2 = SignatureGenerator::computeSpatialSignature(
                        toRect(detections[j]), detections[j].depth, detections[j].conf);

                float dist = cv::EMD(sig1, sig2, cv::DIST_L2, cv::noArray(),
                                     nullptr, cv::noArray());

                costMatrix(i, j) = dist;
            }
            else {
                costMatrix(i, j) = std::numeric_limits<float>::max(); // No match
            }
        }
    }
    return costMatrix;
}