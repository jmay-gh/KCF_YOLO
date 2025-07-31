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
    if (config.removal == UserConfig::STRIKE_BASED) {
        for (auto it = trackers.begin(); it != trackers.end(); ) {

            bool remove = false;

            if (config.occlusion == UserConfig::RELAXED_REMOVAL) {
                // Use different thresholds depending on occlusion state
                if (it->isOccluded) {
                    if (it->consecutiveLoses > occLossThreshold) remove = true;
                }
                else {
                    if (it->consecutiveLoses > lossThreshold) remove = true;
                }
            }
            else {
                // Normal strict removal logic (no relaxed threshold)
                if (it->consecutiveLoses > lossThreshold) remove = true;
            }

            if (remove) it = trackers.erase(it);
            else ++it;
        }
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

        newTracker.confThreshold = trackConfThreshold;
        newTracker.occConfThreshold = occConfThreshold;

        trackers.emplace_back(newTracker);
    }

    // Mark or remove unmatched trackers
    for (auto it = trackers.begin(); it != trackers.end(); ) {
       if (!it->checkMatched()) it->addMiss();
       if (it->consecutiveMisses > nomatchThreshold) it = trackers.erase(it);
       else ++it;
    }
}


void TrackerManager::matchOccludedTrackers(MatchingManager::MatchResult& matchResult,
                                           vector<Segmentation>& detections) {

    // Get occluded trackers
    std::vector<TrackedObject> occTrackers;
    std::vector<int> occTrackersIdx;
    for (auto trackIdx : matchResult.unmatchedTrackers) {
        if (trackers[trackIdx].isOccluded) {
            occTrackers.push_back(trackers[trackIdx]);
            occTrackersIdx.push_back(trackIdx);
        }
    }
    if (occTrackers.empty()) return;

    // Get unmatched detections
    std::vector<Segmentation> unmatchedDetects;
    std::vector<int> unmatchedDetectsIdx;
    for (auto detectIdx : matchResult.unmatchedDetections) {
        unmatchedDetects.push_back(detections[detectIdx]);
        unmatchedDetectsIdx.push_back(detectIdx);
    }

    // Reduce matching threshold and match
    float originalThreshold = config.matchThreshold;
    config.matchThreshold = relaxedMatchThreshold;
    auto occMatchResult = matchFunc(occTrackers, unmatchedDetects);
    config.matchThreshold = originalThreshold;

    // Update matched results
    for (auto match : occMatchResult.matches) {
        // Get global indexes
        int globalTrackIdx = occTrackersIdx[match.first];
        int globalDetIdx = unmatchedDetectsIdx[match.second];
        // Match tracker
        trackers[globalTrackIdx].matchTracker(detections[globalDetIdx], currentFrame);
        // Update matchResult
        matchResult.matches.emplace_back(globalTrackIdx, globalDetIdx);
        matchResult.unmatchedTrackers.erase(globalTrackIdx);
        matchResult.unmatchedDetections.erase(globalDetIdx);
    }
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
    float minVal, maxVal = trackers[0].depth;
    for (auto& tracker : trackers) {
        if (tracker.depth < 0) continue;
        minVal = std::min(minVal, tracker.depth);
        maxVal = std::max(maxVal, tracker.depth);
    }
    return {minVal, maxVal};
}
