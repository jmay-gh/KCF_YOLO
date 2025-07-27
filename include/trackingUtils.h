#pragma once

#include "seg/YOLO11Seg.hpp"

namespace trackingUtils {

    // Converts YOLO bounding boxes to Rects
    inline cv::Rect toRect(const Segmentation& det) {
        return {det.box.x, det.box.y, det.box.width, det.box.height};
    }

    inline cv::Rect toSafeBox(const cv::Rect& box, cv::Mat& frame) {
        return {box & cv::Rect(0, 0, frame.cols, frame.rows)};
    }

    // Converts a Rect to a Segmentation object
    inline BoundingBox toBoundingBox(const cv::Rect& rect) {
        BoundingBox box;
        box.x = rect.x;
        box.y = rect.y;
        box.width = rect.width;
        box.height = rect.height;
        return box;
    }

    // Gets the centre point of a Rect
    inline cv::Point2d rectCenter(const cv::Rect& rect) {
        return cv::Point2d((rect.tl() + rect.br()) * 0.5);
    }

    // Resizes a rect around its center
    inline cv::Rect resizeRect(const cv::Rect& rect, int newWidth, int newHeight) {
        cv::Point2d center = rectCenter(rect);
        return cv::Rect(center.x - newWidth / 2, center.y - newHeight / 2, newWidth, newHeight);
    }

    inline double getDepth(Segmentation det, cv::Mat& depthMap) {
        cv::Scalar meanDepth = cv::mean(depthMap, det.mask);
        return meanDepth[0];
    }

}
