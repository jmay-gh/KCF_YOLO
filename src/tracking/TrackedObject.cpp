#include "../include/tracking/TrackedObject.h"

// Constructor
TrackedObject::TrackedObject(UserConfig& config, Segmentation& detection, cv::Mat& frame, int trackerNum)
        : tracker(config.HOG, config.FIXEDWINDOW, config.MULTISCALE, config.LAB)
{
    // Init tracker
    cv::Rect box = trackingUtils::toRect(detection);
    tracker.init(box, frame);

    // Set tracker ids
    trackerId = trackerNum;
    className = detection.className;
    color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);

    // Set params
    bbox = box;
    depth = detection.depth;
    conf = detection.conf;

    if (config.occlusion == UserConfig::RELAXED_REMOVAL) {
        occRemoval = true;
    }
}


// Tracker states
bool TrackedObject::checkMatched() { return isMatched; }

bool TrackedObject::checkOccluded() { return isOccluded; }

void TrackedObject::addMiss() { consecutiveMisses++; }

void TrackedObject::addLoss() { consecutiveLoses++; }

//bool TrackedObject::checkMisses() { return consecutiveMisses >= 5; }

//bool TrackedObject::checkLoss() { return consecutiveLoses >= 12; }

// Match tracker with new detection
void TrackedObject::matchTracker(Segmentation& detection, cv::Mat& frame) {

    // Get detection box
    cv::Rect box = toSafeBox(toRect(detection), frame);

    // Init tracker with detection
    tracker.init(box, frame);

    // Set initial confidence, bounding box and depth
    conf = detection.conf;
    depth = detection.depth;
    bbox = box;

    // Reset matched state
    consecutiveMisses = 0;
    isMatched = true;
}

// Update the tracker for new frame
void TrackedObject::updateTracker(cv::Mat& frame) {

    // Update tracker conf and bounding box
    bbox = toSafeBox(tracker.update(frame), frame);
    conf = tracker.best_peak_value;

    // Check for tracker loss
    if (!isOccluded && conf < confThreshold) consecutiveLoses++;
    else if (occRemoval && isOccluded && conf < occConfThreshold) consecutiveLoses++;
    else consecutiveLoses = 0;

    // Reset unmatched state
    isMatched = false;
}

// Draw the tracker on the frame
void TrackedObject::drawTracker(cv::Mat& frame, float minDepth, float maxDepth) {

    // Draw depth bounding box
    float normDepth = std::clamp((depth-minDepth)/(maxDepth-minDepth), 0.0f, 1.0f);

    int hue = static_cast<int>(60.0f * (1.0f - normDepth));
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
    putText(frame, std::to_string(trackerId) + ", " + className + ", conf: " + std::to_string(conf),
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
