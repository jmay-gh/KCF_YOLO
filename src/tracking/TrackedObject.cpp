#include <printf.h>
#include "../include/tracking/TrackedObject.h"
#include "trackingUtils.h"

using namespace trackingUtils;
using namespace cv;

// Constructor
TrackedObject::TrackedObject(TrackerConfig config, cv::Rect bbox, std::string label, cv::Mat frame, int trackerNum, double depth)
        : tracker(config.HOG, config.FIXEDWINDOW, config.MULTISCALE, config.LAB),
        bbox(bbox), label(label), trackerNum(trackerNum), depth(depth)
{
    tracker.init(bbox, frame);
    tracker.update(frame);
    conf = tracker.best_peak_value;

    isOccluded = false;
    matchedDetector = true;

    consecutiveLoss = 0;
    consecutiveFailures = 0;

    // Random colour assignment
    color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
}


// Set matched from detector
void TrackedObject::setMatched() { matchedDetector = true; }
// Set unmatched from detector
void TrackedObject::setUnmatched() { matchedDetector = false; }
// Get the result of the match
bool TrackedObject::checkMatched() { return matchedDetector; }


// Update a failure to detect a checker
void TrackedObject::setFailure() { consecutiveFailures++; }
// Reset failures to 0
void TrackedObject::resetFailures() { consecutiveFailures = 0; }
// Check if the failure threshold reached
bool TrackedObject::checkFailures() { return consecutiveFailures >= 15; }

// Match tracker
void TrackedObject::matchTracker(cv::Rect box, double meanDepth, const cv::Mat& frame) {
    // Update bbox and frame
    tracker.init(box, frame);
    bbox = box;
    depth = meanDepth;
    // Reset consecutive missed detections and set as matched
    resetFailures();
    setMatched();
}

void TrackedObject::updateTracker(const cv::Mat& frame) {
    bbox = tracker.update(frame);
    conf = tracker.best_peak_value;
}


// Draw tracker
void TrackedObject::draw(cv::Mat& frame, float minDepth, float maxDepth) {

    // Draw depth bounding box
    float normDepth = std::clamp((depth-minDepth)/(maxDepth-minDepth), 0.0f, 1.0f);

    int hue = static_cast<int>(120.0f * (1.0f - normDepth));  // 120 (green) to 0 (red)
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Scalar depthColor = bgr.at<cv::Vec3b>(0, 0);

    cv::Rect depthBox(bbox.x+4, bbox.y+4, bbox.width-8, bbox.height-8);
    rectangle(frame, depthBox, depthColor, 4);

    // Draw main bounding box
    rectangle(frame, bbox, color, 2);
    cv::Point point = bbox.tl();
    point.y -= 5;
    putText(frame, std::to_string(trackerNum) + ", " + label + ", conf: " + std::to_string(conf),
            point, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

    // Draw tracker confidence
    cv::Point depthPoint = bbox.tl();
    depthPoint.y += bbox.height + 20;
    if (isOccluded) {
        putText(frame, "occluded", depthPoint,
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    } else {
        putText(frame, "depth: " + std::to_string(depth), depthPoint,
                cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
    }
}
