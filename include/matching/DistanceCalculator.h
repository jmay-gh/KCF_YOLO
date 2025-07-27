//
// Created by Jack May on 25/7/2025.
//

#ifndef KCF_YOLO_DISTANCECALCULATOR_H
#define KCF_YOLO_DISTANCECALCULATOR_H

#pragma once

#include <opencv2/opencv.hpp>
#include "trackingUtils.h"

using namespace trackingUtils;

namespace DistanceCalculator {

    inline float iou(const cv::Rect& a, const cv::Rect& b) {
        float intersectionArea = (a & b).area();
        float unionArea = a.area() + b.area() - intersectionArea;
        return intersectionArea / unionArea;
    }

    inline float inverseIou(const cv::Rect& a, const cv::Rect& b) {
        return 1.0f - iou(a, b);
    }

    inline float euclidean(const cv::Rect& a, const cv::Rect& b) {
        cv::Point2f ca = rectCenter(a);
        cv::Point2f cb = rectCenter(b);
        return static_cast<float>(cv::norm(ca - cb));
    }

    inline float ios(const cv::Rect& a, const cv::Rect& b) {
        float intersectionArea = (a & b).area();
        float minArea = std::min(a.area(), b.area());
        return intersectionArea / minArea;
    }
}

#endif //KCF_YOLO_DISTANCECALCULATOR_H
