//#ifndef KCF_YOLO_COSTMATRIXBUILDER_H
//#define KCF_YOLO_COSTMATRIXBUILDER_H

#pragma once

#include "matching/DistanceCalculator.h"
#include "matching/SignatureGenerator.h"
#include "hungarian_algo/matrix.h"
#include "trackingUtils.h"

class CostMatrixBuilder {

public:
    using DistanceFunc = std::function<float(const cv::Rect&, const cv::Rect&)>;

    static Matrix<float> buildCostMatrix(
            const std::vector<TrackedObject>& trackers,
            const std::vector<Segmentation>& detections,
            DistanceFunc distanceFunc
    );

    static Matrix<float> buildFlowMatrix(
            const std::vector<TrackedObject>& trackers,
            const std::vector<Segmentation>& detections,
            cv::Mat& frame,
            UserConfig& config,
            float matchingThreshold
    );
};

//#endif //KCF_YOLO_COSTMATRIXBUILDER_H
