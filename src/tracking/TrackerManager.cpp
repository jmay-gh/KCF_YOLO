#include "../include/tracking/TrackerManager.h"

using namespace std;
using namespace trackingUtils;

TrackerManager::TrackerManager(const UserConfig& config, cv::Mat& frame)
    : matchingManager(config, frame), config(config), currentFrame(frame) {

    using enum UserConfig::AssociationMethod;
    if (config.association == NEAREST_NEIGHBOUR) {
        matchFunc = [this](auto& trackers, const auto& detections) {
            return matchingManager.matchNN(trackers, detections);
        };
    } else if (config.association == HUNGARIAN_ALGORITHM) {
        matchFunc = [this](auto& trackers, const auto& detections) {
            return matchingManager.matchHungarian(trackers, detections);
        };
    } else if (config.association == GROUND_MOVERS_DISTANCE) {
        matchFunc = [this](auto& trackers, const auto& detections) {
            return matchingManager.matchEMD(trackers, detections);
        };
    } else {
        throw std::invalid_argument("Unsupported association method");
    }

    trackConfThreshold = 0.3f;
    detectConfThreshold = 0.3f;
    lossThreshold = 12;
}

void TrackerManager::updateTrackers() {
    for (auto& tracker : trackers) {
        tracker.updateTracker(currentFrame);

        // CHECK FOR OCCLUSIONS
        tracker.isOccluded = false;

        if (config.occlusion != UserConfig::NO_OCCLUSION) {
            for (auto& otherTracker: trackers) {
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
    }

    // REMOVE TRACKERS
    for (auto it = trackers.begin(); it != trackers.end(); ) {
        bool remove = false;
        if (config.removal == UserConfig::THRESHOLD) {
            remove = config.occlusion == UserConfig::RELAXED_REMOVAL
                     ? (it->conf < trackConfThreshold && !it->isOccluded)
                     : (it->conf < trackConfThreshold);
        }
        else if (config.removal == UserConfig::STRIKE_BASED) {
            remove = config.occlusion == UserConfig::RELAXED_REMOVAL
                     ? (it->checkLoss() && !it->isOccluded)
                     : (it->checkLoss());
        }
        if (remove) it = trackers.erase(it);
        else ++it;
    }
}


void TrackerManager::matchTrackers(vector<Segmentation>& detections) {

    // Match trackers to new detections
    auto matchResult = matchFunc(trackers, detections);
    for (auto& [trackIdx, detIdx] : matchResult.matches) {
        trackers[trackIdx].matchTracker(detections[detIdx], currentFrame);
    }

    // Match occluded trackers with new detections (relaxed matching)
    if (config.occlusion == UserConfig::RELAXED_MATCHING) {
        matchOccludedTrackers(matchResult, detections);
    }

    // Add new trackers for unmatched detections
    for (auto& detIdx : matchResult.unmatchedDetections) {
        TrackedObject newTracker(config, detections[detIdx], currentFrame, trackerCounter++);
        trackers.emplace_back(newTracker);
    }

    // Mark or remove unmatched trackers
    for (auto it = trackers.begin(); it != trackers.end(); ) {
       if (!it->checkMatched()) it->addMiss();
       if (it->checkMisses()) it = trackers.erase(it);
       else ++it;
    }
}


void TrackerManager::matchOccludedTrackers(MatchingManager::MatchResult& matchResult,
                                           vector<Segmentation>& detections) {
//    // Try to match occluded trackers with new detections
//    set<int> matchedTrackers;
//    set<int> matchedDetections;
//
//    for (auto& trackIdx : matchResult.unmatchedTrackers) {
//
//        if (!trackers[trackIdx].isOccluded) continue;
//
//        for (auto& detIdx : matchResult.unmatchedDetections) {
//
//            if (matchedDetections.count(detIdx)) continue; // Skip if already matched
//
//            float distance = matchingManager.euclidean(trackers[trackIdx].bbox, toRect(detections[detIdx]));
//
//            float distanceThreshold = std::sqrt(currentFrame.cols * currentFrame.cols +
//                    currentFrame.rows * currentFrame.rows) * 0.05f;
//
//            if (distance <= distanceThreshold) {
//                trackers[trackIdx].matchTracker(detections[detIdx], currentFrame);
//                matchResult.matches.emplace_back(trackIdx, detIdx);
//                matchedTrackers.insert(trackIdx);
//                matchedDetections.insert(detIdx);
//                cout << "Matched an occluded tracker" << endl;
//                break;
//            }
//        }
//    }
//
//    for (int trackIdx : matchedTrackers) {
//        if (matchResult.unmatchedTrackers.count(trackIdx))
//            matchResult.unmatchedTrackers.erase(trackIdx);
//    }
//
//    for (int detIdx : matchedDetections) {
//        if (matchResult.unmatchedDetections.count(detIdx))
//            matchResult.unmatchedDetections.erase(detIdx);
//    }
}

// Draw all trackers
void TrackerManager::drawTrackers(cv::Mat& frame) {
    pair<float, float> depths = getMinMaxDepth();
    for (auto& tracker : trackers) tracker.drawTracker(frame, depths.first, depths.second);
}

// Write tracker information to output file
void TrackerManager::outputTrackers(std::ofstream& out, int frameIdx) {
    for (const auto& tracker : trackers) {
        out << frameIdx + 1 << ","          // Frame number
            << tracker.trackerId << ","    // Tracker ID
            << tracker.bbox.x << ","        // x
            << tracker.bbox.y << ","        // y
            << tracker.bbox.width << ","    // width
            << tracker.bbox.height << ","   // height
            << 1 << "," << -1 << "," << -1 << "," << -1 << std::endl;
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
