#pragma once

#include "seg/YOLO11Seg.hpp"

using namespace cv;

namespace trackingUtils {

    // Converts YOLO bounding boxes to Rects
    inline Rect toRect(const Segmentation& det) {
        return Rect(det.box.x, det.box.y, det.box.width, det.box.height);
    }

    // Converts a Rect to a Segmentation object
    inline BoundingBox toBoundingBox(const Rect& rect) {
        BoundingBox box;
        box.x = rect.x;
        box.y = rect.y;
        box.width = rect.width;
        box.height = rect.height;
        return box;
    }

    // Gets the centre point of a Rect
    inline Point2d rectCenter(const Rect& rect) {
        return Point2d((rect.tl() + rect.br()) * 0.5);
    }

    // Resizes a rect around its center
    inline Rect resizeRect(const Rect& rect, int newWidth, int newHeight) {
        Point2d center = rectCenter(rect);
        return Rect(center.x - newWidth / 2, center.y - newHeight / 2, newWidth, newHeight);
    }

    inline double getDepth(Segmentation det, Mat& depthMap) {
        cv::Scalar meanDepth = cv::mean(depthMap, det.mask);
        return meanDepth[0];
    }

}
