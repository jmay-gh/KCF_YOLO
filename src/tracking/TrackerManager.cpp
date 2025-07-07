#include "../include/tracking/TrackerManager.h"

using namespace cv;
using namespace std;
using namespace trackingUtils;

TrackerManager::TrackerManager(const TrackerConfig& config, const vector<string>& classNames, Mat& currentFrame)
    : matchingManager(config), config(config), classNames(classNames), currentFrame(currentFrame) { }

void TrackerManager::updateTrackers(const Mat& frame) {
    for (auto& tracker : trackers) { tracker.updateTracker(frame); }

    // Freeze trackers that are occluded
    for (auto& tracker : trackers) {
        for (auto& otherTracker : trackers) {
            if (&tracker == &otherTracker || // Skip if same tracker
            tracker.depth > otherTracker.depth || // Skip if tracker is closer
            otherTracker.isOccluded) continue; // Skip if other tracker is already occluded

            cv::Point center = trackingUtils::rectCenter(tracker.bbox);
            if (otherTracker.bbox.contains(center)) {
                tracker.isOccluded = true;
                cout << "Tracker " << tracker.trackerNum << " is occluded by Tracker " << otherTracker.trackerNum << endl;
                break;
            }
            else tracker.isOccluded = false;
        }
        if (tracker.conf < 0.3) tracker.consecutiveLoss++;
        else tracker.consecutiveLoss = 0;
    }

    // Remove trackers that have low confidence
    for (auto it = trackers.begin(); it != trackers.end(); ) {
        if (it->consecutiveLoss >= 15 && !it->isOccluded) it = trackers.erase(it);
        else ++it;
    }
}

void TrackerManager::updateTrackersWithDetections(const Mat& frame, const vector<Segmentation>& detections, Mat depthMap) {
    currentFrame = frame;
    // Match detections to trackers
    auto matchResult = matchDetections(detections);
    for (auto& [trackIdx, detIdx] : matchResult.matches) {
        Rect bbox = toRect(detections[detIdx]);
        trackers[trackIdx].matchTracker(bbox, getDepth(detections[detIdx], depthMap), frame);
    }

    // Try to match occluded trackers with new detections
    set<int> matchedTrackers;
    set<int> matchedDetections;
    for (auto& trackIdx : matchResult.unmatchedTrackers) {
        if (!trackers[trackIdx].isOccluded) continue;
        for (auto& detIdx : matchResult.unmatchedDetections) {
            if (matchedDetections.count(detIdx)) continue; // Skip if already matched
            float distance = matchingManager.euclidean(trackers[trackIdx].bbox, toRect(detections[detIdx]));
            float maxDistance = frame.cols * 0.1f;
            if (distance <= maxDistance) {
                trackers[trackIdx].matchTracker(toRect(detections[detIdx]),
                                                getDepth(detections[detIdx], depthMap), frame);
                matchResult.matches.emplace_back(trackIdx, detIdx);

                matchedTrackers.insert(trackIdx);
                matchedDetections.insert(detIdx);

                cout << "Matched an occluded tracker" << endl;

                break; // Break after matching one detection
            }
        }
    }
    for (int trackIdx : matchedTrackers) {
        if (matchResult.unmatchedTrackers.count(trackIdx))
            matchResult.unmatchedTrackers.erase(trackIdx);
    }
    for (int detIdx : matchedDetections) {
        if (matchResult.unmatchedDetections.count(detIdx))
            matchResult.unmatchedDetections.erase(detIdx);
    }

    // Create new trackers for unmatched detections
    for (auto& detIdx : matchResult.unmatchedDetections) {
        Rect bbox = toRect(detections[detIdx]);
        TrackedObject newTracker(config, bbox,classNames[detections[detIdx].classId], frame,
                                 trackerCounter++, getDepth(detections[detIdx], depthMap));
        trackers.emplace_back(newTracker);
    }

    // Remove trackers that have not been matched for a while
    for (auto it = trackers.begin(); it != trackers.end(); ) {
       if (!it->checkMatched()) it->setFailure();
       if (it->checkFailures() && !it->isOccluded) it = trackers.erase(it);
       else ++it;
    }
}

void TrackerManager::drawTrackers(Mat& frame) {
    pair<float, float> depths = getMinMaxDepth();
    for (auto& tracker : trackers) tracker.draw(frame, depths.first, depths.second);
}

MatchingManager::MatchResult TrackerManager::matchDetections(const vector<Segmentation>& detections) {
    MatchingManager::MatchResult matchResult;
    if (config.association == TrackerConfig::NEAREST_NEIGHBOUR) {
        matchResult = matchingManager.matchNN(trackers, detections);
    }
    else if (config.association == TrackerConfig::HUNGARIAN_ALGORITHM) {
        matchResult = matchingManager.matchHungarian(trackers, detections);
    }
    else if (config.association == TrackerConfig::GROUND_MOVERS_DISTANCE) {
        matchResult = matchingManager.matchEMD(trackers, detections, currentFrame);
    }
    return matchResult;
}

double TrackerManager::getDepth(Segmentation det, Mat depthMap) {
    cv::Scalar meanDepth = cv::mean(depthMap, det.mask);
    return meanDepth[0];
}

pair<float, float> TrackerManager::getMinMaxDepth() {
    float minVal, maxVal = 0.0f;
    for (auto& tracker : trackers) {
        if (tracker.depth < 0) continue;
        if (minVal > tracker.depth || minVal < 0) minVal = tracker.depth;
        if (maxVal < tracker.depth) maxVal = tracker.depth;
    }
    return {minVal, maxVal}; // or maxVal, depending on your needs
}
