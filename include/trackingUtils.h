#pragma once

#include "seg/YOLO11Seg.hpp"

using namespace cv;

namespace trackingUtils {

    // Converts YOLO bounding boxes to Rects
    inline Rect toRect(const Segmentation& det) {
        return Rect(det.box.x, det.box.y, det.box.width, det.box.height);
    }

    // Gets the centre point of a Rect
    inline Point2d rectCenter(const Rect& rect) {
        return Point2d((rect.tl() + rect.br()) * 0.5);
    }
}
