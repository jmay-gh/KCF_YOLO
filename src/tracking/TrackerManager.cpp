#include "../include/tracking/TrackerManager.h"

using namespace cv;
using namespace std;
using namespace trackingUtils;

TrackerManager::TrackerManager(const TrackerConfig& config, const vector<string>& classNames, Mat& currentFrame)
    : matchingManager(config), config(config), classNames(classNames), currentFrame(currentFrame)
{
    trackConfThreshold = 0.3f;
    detectConfThreshold = 0.3f;
    unmatchedThreshold = 12;
}

void TrackerManager::updateTrackers(const Mat& frame) {
    for (auto& tracker : trackers) {
        tracker.updateTracker(frame);

        // CHECK FOR OCCLUSIONS
        tracker.isOccluded = false;
        if (config.occlusion != TrackerConfig::NO_OCCLUSION) {
            for (auto &otherTracker: trackers) {
                if (&tracker == &otherTracker || // Skip if same tracker
                    tracker.depth > otherTracker.depth || // Skip if tracker is closer
                    otherTracker.isOccluded)
                    continue; // Skip if other tracker is already occluded

                cv::Point center = trackingUtils::rectCenter(tracker.bbox);
                if (otherTracker.bbox.contains(center)) {
                    tracker.isOccluded = true;
                    break;
                }
            }
        }

        // CHECK FOR REMOVAL
        if (config.removal == TrackerConfig::STRIKE_BASED) {
            if (tracker.conf < trackConfThreshold) tracker.consecutiveLoss++;
            else tracker.consecutiveLoss = 0;
        }
    }

    // REMOVE TRACKERS
    for (auto it = trackers.begin(); it != trackers.end(); ) {
        bool remove = false;
        if (config.removal == TrackerConfig::THRESHOLD) {
            remove = config.occlusion == TrackerConfig::RELAXED_REMOVAL
                     ? (it->conf < trackConfThreshold && !it->isOccluded)
                     : (it->conf < trackConfThreshold);
        } else if (config.removal == TrackerConfig::STRIKE_BASED) {
            remove = config.occlusion == TrackerConfig::RELAXED_REMOVAL
                     ? (it->consecutiveLoss > unmatchedThreshold && !it->isOccluded)
                     : (it->consecutiveLoss > unmatchedThreshold);
        }
        if (remove) it = trackers.erase(it);
        else ++it;
    }
}

void TrackerManager::updateTrackersWithDetections(const Mat& frame, vector<Segmentation>& detections) {
    currentFrame = frame;
    matchingManager.currentFrame = frame;

    // Match detections to trackers
    auto matchResult = matchDetections(detections);
    for (auto& [trackIdx, detIdx] : matchResult.matches) {
        Rect bbox = toRect(detections[detIdx]);
        trackers[trackIdx].matchTracker(bbox, getDepth(detections[detIdx], depthMap), frame);
    }

    if (config.occlusion == TrackerConfig::RELAXED_MATCHING) {
        matchOccluded(matchResult, detections, depthMap);
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


void TrackerManager::matchOccluded(MatchingManager::MatchResult& matchResult,
                                   const vector<Segmentation>& detections,
                                   Mat depthMap) {
        // Try to match occluded trackers with new detections
    set<int> matchedTrackers;
    set<int> matchedDetections;
    for (auto& trackIdx : matchResult.unmatchedTrackers) {
        if (!trackers[trackIdx].isOccluded) continue;
        for (auto& detIdx : matchResult.unmatchedDetections) {
            if (matchedDetections.count(detIdx)) continue; // Skip if already matched

            float distance = matchingManager.euclidean(trackers[trackIdx].bbox, toRect(detections[detIdx]));
            float distanceThreshold = std::sqrt(currentFrame.cols * currentFrame.cols +
                    currentFrame.rows * currentFrame.rows) * 0.05f;

            if (distance <= distanceThreshold) {
                trackers[trackIdx].matchTracker(toRect(detections[detIdx]),
                                                getDepth(detections[detIdx], depthMap), currentFrame);
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


void TrackerManager::outputTrackers(std::ofstream& out, int frameIdx) {
    for (const auto& tracker : trackers) {
        out << frameIdx + 1 << ","                // Frame number
             << tracker.trackerNum << ","             // Tracker ID
             << tracker.bbox.x << ","              // x
             << tracker.bbox.y << ","              // y
             << tracker.bbox.width << ","              // width
             << tracker.bbox.height << ","              // height
             << 1 << ","          // Confidence score (1.0 if N/A)
             << -1 << ","                   // Optional (e.g., 3D pos) – set to -1
             << -1 << ","                   // Optional
             << -1 << std::endl;            // Optional
    }
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
