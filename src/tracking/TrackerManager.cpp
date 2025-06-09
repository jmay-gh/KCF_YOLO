#include "../include/tracking/TrackerManager.h"

using namespace cv;
using namespace std;

TrackerManager::TrackerManager(const TrackerConfig& config, const vector<string>& classNames)
    : config(config), classNames(classNames) { }

// PUBLIC METHODS
void TrackerManager::updateTrackers(const Mat& frame, const vector<Detection>& detections, int frameIdx) {
    if (!detections.empty()) {
        // Match detections to trackers
        auto matchResult = matchDetections(detections);
        for (auto& [trackIdx, detIdx] : matchResult) {
            trackers[trackIdx].matchTracker(toRect(detections[detIdx]), frame);
        }
        // Create new trackers for unmatched detections
        auto unmatched = findUnmatchedDetections(matchResult, detections.size());
        for (int detIdx : unmatched) {
            TrackedObject newTracker(config, toRect(detections[detIdx]),
                                     classNames[detections[detIdx].classId], frame, trackerCounter++);
            trackers.emplace_back(newTracker);
        }
        // Remove trackers that have not been matched for a while
        for (auto it = trackers.begin(); it != trackers.end(); ) {
            if (!it->checkMatched()) it->setFailure();
            if (it->checkFailures()) it = trackers.erase(it);
            else ++it;
        }
    }
    else {
        for (auto& tracker : trackers) {
            tracker.bbox = tracker.tracker.update(frame);
        }
    }
}

void TrackerManager::drawTrackers(Mat& frame) {
    for (auto& tracker : trackers) tracker.draw(frame);
}

// PRIVATE METHODS
Rect TrackerManager::toRect(const Detection& det) {
    return Rect(det.box.x, det.box.y, det.box.width, det.box.height);
}

float TrackerManager::calculateIOU(const Rect& a, const BoundingBox& b) {
    float x1 = max(a.x, b.x);
    float y1 = max(a.y, b.y);
    float x2 = min(a.x + a.width, b.x + b.width);
    float y2 = min(a.y + a.height, b.y + b.height);

    float width = max(0.0f, x2 - x1);
    float height = max(0.0f, y2 - y1);
    float inter = width * height;
    float unionArea = a.area() + b.area() - inter;

    return unionArea > 0 ? inter / unionArea : 0.0f;
}

vector<pair<int, int>> TrackerManager::matchDetections(const vector<Detection>& detections) {
    size_t nTrackers = trackers.size();
    size_t nDetections = detections.size();

    vector<pair<int, int>> matches;
    if (nTrackers == 0 || nDetections == 0) return {};

    Matrix<float> costMatrix = solveCostMatrix(detections);

    float iouThreshold = 0.15f;
    for (size_t i = 0; i < nTrackers; ++i) {
        trackers[i].setUnmatched();
        for (size_t j = 0; j < nDetections; ++j) {
            if (costMatrix(i, j) == 0.0f) {
                float iou = calculateIOU(trackers[i].bbox, detections[j].box);
                if (iou >= iouThreshold) {
                    matches.emplace_back(i, j);
                    trackers[i].setMatched();
                    break;
                }
            }
        }
    }
    return matches;
}


Matrix<float> TrackerManager::solveCostMatrix(vector<Detection> detections) {
    // Create cost matrix
    size_t nTrackers = trackers.size();
    size_t nDetections = detections.size();
    Matrix<float> costMatrix(nTrackers, nDetections);
    // Populate cost matrix
    for (size_t i = 0; i < nTrackers; ++i) {
        for (size_t j = 0; j < nDetections; ++j) {
            float iou = calculateIOU(trackers[i].bbox, detections[j].box);
            costMatrix(i, j) = 1.0f - iou;
        }
    }
    // Solve cost matrix using Munkres algorithm
    Munkres<float> munkres;
    munkres.solve(costMatrix);
    return costMatrix;
}


vector<int> TrackerManager::findUnmatchedDetections(const vector<pair<int, int>>& matches, int totalDetections) {
    set<int> matched;
    for (auto& m : matches) matched.insert(m.second);

    vector<int> unmatched;
    for (int i = 0; i < totalDetections; ++i) {
        if (matched.find(i) == matched.end()) unmatched.push_back(i);
    }
    return unmatched;
}

