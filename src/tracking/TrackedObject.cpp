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
    if (config.auxType == UserConfig::AuxType::DEPTH) {
        depthOn = true;
    }

    if (config.occlusion == UserConfig::RELAXED_REMOVAL ||
        config.occlusion == UserConfig::REMOVAL_AND_MATCHING) {
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

    updateMotionHistory();

    // Determine which threshold to use
    float activeThreshold = confThreshold;
    if (occRemoval && isOccluded && occConfThreshold != confThreshold) {
        // Only switch threshold if different
        activeThreshold = occConfThreshold;
    }
    // Loss check — identical if thresholds are the same
    if (conf < activeThreshold) consecutiveLoses++;
    else consecutiveLoses = 0;

    // Reset unmatched state
    isMatched = false;
}


// Draw the tracker on the frame
void TrackedObject::drawTracker(cv::Mat& frame, float minDepth, float maxDepth) {

    // Draw main bounding box
    rectangle(frame, bbox, color, 2);
    cv::Point point = bbox.tl();
    point.y -= 5;
    putText(frame, std::to_string(trackerId) + ", " + className + ", conf: " + std::to_string(conf),
            point, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

    // Draw tracker confidence
    if (depthOn) {
        // Draw depth bounding box
        float normDepth = std::clamp((depth-minDepth)/(maxDepth-minDepth), 0.0f, 1.0f);
        int hue = static_cast<int>(60.0f * (1.0f - normDepth));
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::Scalar depthColor = bgr.at<cv::Vec3b>(0, 0);

        cv::Rect depthBox(bbox.x+4, bbox.y+4, bbox.width-8, bbox.height-8);
        rectangle(frame, depthBox, depthColor, 4);

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
}



void TrackedObject::updateMotionHistory() {
    cv::Point2f center(bbox.x + bbox.width * 0.5f,
                       bbox.y + bbox.height * 0.5f);
    positionHistory.push_back(center);
    if (positionHistory.size() > motionWindowSize) {
        positionHistory.pop_front();
    }

    // Refresh motion subspace every update
    motionSubspace = computeMotionSubspace(positionHistory);
}


cv::Mat TrackedObject::buildHankel(const std::vector<float>& seq, int rows) const {
    int cols = seq.size() - rows + 1;
    cv::Mat H(rows, cols, CV_32F);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            H.at<float>(r, c) = seq[r + c];
        }
    }
    return H;
}

cv::Mat TrackedObject::computeMotionSubspace(const std::deque<cv::Point2f>& history) const {
    if (history.size() < hankelRows + 1)
        return cv::Mat();

    // Split into X and Y sequences
    std::vector<float> xs, ys;
    xs.reserve(history.size());
    ys.reserve(history.size());
    for (const auto& p : history) {
        xs.push_back(p.x);
        ys.push_back(p.y);
    }

    // Build Hankel matrices for X and Y
    cv::Mat Hx = buildHankel(xs, hankelRows);
    cv::Mat Hy = buildHankel(ys, hankelRows);

    // Concatenate
    cv::Mat H;
    cv::vconcat(Hx, Hy, H);

    // SVD
    cv::SVD svd(H, cv::SVD::MODIFY_A);
    int rank = 2; // For constant velocity model
    cv::Mat U_reduced = svd.u(cv::Rect(0, 0, rank, svd.u.rows)).clone();

    return U_reduced;
}

float TrackedObject::motionConsistencyScore(const cv::Point2f& candidate) const {
    // Require enough history and valid motion subspace
    if (positionHistory.size() < 5 || motionSubspace.empty())
        return 0.0f;

    // Create hypothetical trajectory including candidate
    auto histWithCandidate = positionHistory;
    histWithCandidate.push_back(candidate);
    if (histWithCandidate.size() > motionWindowSize)
        histWithCandidate.pop_front();

    // Compute subspace for the new trajectory
    cv::Mat U_new = computeMotionSubspace(histWithCandidate);
    if (U_new.empty()) return 0.0f;

    // Compute projection matrices
    cv::Mat P_old = motionSubspace * motionSubspace.t(); // Projection onto current subspace
    cv::Mat P_new = U_new * U_new.t();                   // Projection onto candidate subspace

    // Frobenius norm of the difference as a distance metric
    float diffNorm = static_cast<float>(cv::norm(P_old - P_new, cv::NORM_L2));

    // Normalize score to [0, 1] (optional scaling factor can be tuned)
    // Maximum possible norm difference for rank-r orthonormal subspaces is sqrt(2*r)
    float maxNorm = std::sqrt(2.0f * motionSubspace.cols);
    float score = std::min(1.0f, diffNorm / maxNorm);

    // Smaller score = more consistent; invert to match "cost" semantics if needed
    return score;
}

cv::Point2f TrackedObject::predictNextPosition() const {
    if (positionHistory.size() < 2) {
        // Not enough history, just return current center
        return cv::Point2f(bbox.x + bbox.width * 0.5f,
                           bbox.y + bbox.height * 0.5f);
    }

    cv::Point2f p1 = positionHistory[positionHistory.size() - 2];
    cv::Point2f p2 = positionHistory.back();
    cv::Point2f velocity = p2 - p1;

    return p2 + velocity; // predicted next position
}

