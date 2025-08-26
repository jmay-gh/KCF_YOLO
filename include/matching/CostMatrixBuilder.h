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
            std::vector<TrackedObject>& trackers,
            const std::vector<Segmentation>& detections,
            DistanceFunc distanceFunc,
            cv::Mat& frame,
            UserConfig& config,
            float matchingThreshold
    );

    static Matrix<float> buildCMEMDMatrix(
            const std::vector<TrackedObject>& trackers,
            const std::vector<Segmentation>& detections,
            cv::Mat& frame,
            UserConfig& config,
            float matchingThreshold
    );

    static Matrix<float> buildFMEMDMatrix(
            const std::vector<TrackedObject>& trackers,
            const std::vector<Segmentation>& detections,
            cv::Mat& frame,
            UserConfig& config,
            float matchingThreshold);

    static std::pair<float,float> getMinMaxDepthDet(
            const std::vector<TrackedObject>& trackers,
            const std::vector<Segmentation>& detections);

};




//#endif //KCF_YOLO_COSTMATRIXBUILDER_H
