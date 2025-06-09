#include "../include/tracking/TrackedObject.h"

// Constructor
TrackedObject::TrackedObject(TrackerConfig config, cv::Rect bbox, std::string label, cv::Mat frame, int trackerNum)
        : tracker(config.HOG, config.FIXEDWINDOW, config.MULTISCALE, config.LAB),
        bbox(bbox), label(label), trackerNum(trackerNum)
{
    tracker.init(bbox, frame);
    matchedDetector = true;
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
bool TrackedObject::checkFailures() { return consecutiveFailures >= 7; }


// Match tracker
void TrackedObject::matchTracker(cv::Rect box, cv::Mat frame) {
    // Update bbox and frame
    tracker.init(box, frame);
    bbox = box;
    // Reset consecutive missed detections and set as matched
    resetFailures();
    setMatched();
}


// Draw tracker
void TrackedObject::draw(cv::Mat& frame) {
    rectangle(frame, bbox, color, 2);
    cv::Point point = bbox.tl();
    point.y -= 5;
    putText(frame, std::to_string(trackerNum) + " " + label, point, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
}
