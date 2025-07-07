#include "../include/matching/MatchingManager.h"
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;
using namespace trackingUtils;

MatchingManager::MatchingManager(const TrackerConfig& config)
        : config(config) { }

double MatchingManager::euclidean(const cv::Rect& aRect, const cv::Rect& bRect) {
    Point2d a = rectCenter(aRect); Point2d b = rectCenter(bRect);
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

double MatchingManager::iou(const cv::Rect& a, const cv::Rect& b) {
    int areaIntersection = (a & b).area();
    int areaUnion = a.area() + b.area() - areaIntersection;
    return static_cast<double>(areaIntersection) / areaUnion;
}

double MatchingManager::inverseIou(const cv::Rect& a, const cv::Rect& b) {
    return 1.0f - iou(a, b);
}

// NEAREST NEIGHBOUR ALGO
MatchingManager::MatchResult MatchingManager::matchNN(vector<TrackedObject>& trackers,
                                               const vector<Segmentation>& detections) {

    MatchResult matchResult = setMatchResult(trackers.size(), detections.size());
    if (trackers.size() == 0 || detections.size() == 0) return matchResult;

    for (size_t i = 0; i < trackers.size(); ++i) {
        int bestMatch = -1;
        float bestDistance = numeric_limits<float>::max();
        for (size_t j = 0; j < detections.size(); ++j) {

            if (!matchResult.unmatchedDetections.count(j)) continue;

            float distance;
            if (config.distance == TrackerConfig::DistanceType::EUCLIDEAN) {
                distance = euclidean(trackers[i].bbox, toRect(detections[j]));
            }
            else distance = inverseIou(trackers[i].bbox, toRect(detections[j]));
            // Assign new match
            if (distance < bestDistance) {
                bestMatch = j;
                bestDistance = distance;
            }
        }

        if (bestMatch != -1) {
            // Thresholding the match
            if ((config.distance == TrackerConfig::DistanceType::EUCLIDEAN && bestDistance <= 50.0f) ||
                (config.distance == TrackerConfig::DistanceType::IOU && (1.0f - bestDistance) >= 0.5f)) {
                matchResult.matches.emplace_back(i, bestMatch);
                matchResult.unmatchedTrackers.erase(i);
                matchResult.unmatchedDetections.erase(bestMatch);
                trackers[i].setMatched();
            }
        }
    }
    return matchResult;
}

// HUNGARIAN ALGO
MatchingManager::MatchResult MatchingManager::matchHungarian(vector<TrackedObject>& trackers,
                                     const vector<Segmentation>& detections) {
    MatchResult matchResult = setMatchResult(trackers.size(), detections.size());
    if (trackers.size() == 0 || detections.size() == 0) return matchResult;
    // Create and solve cost matrix
    Matrix<float> costMatrix;
    costMatrix = computeMatrix(trackers, detections);
    Munkres<float> munkres;
    munkres.solve(costMatrix);

    // Assign matches from cost matrix
    for (int i = 0, n = trackers.size(); i < n; ++i) {
        trackers[i].setUnmatched();
        for (int j = 0, m = detections.size(); j < m; ++j) {
            if (costMatrix(i, j) == 0.0f) {
                float iouResult = iou(trackers[i].bbox, toRect(detections[j]));
                if (iouResult >= 0.5f) {
                    matchResult.matches.emplace_back(i, j);
                    matchResult.unmatchedTrackers.erase(i);
                    matchResult.unmatchedDetections.erase(j);
                    trackers[i].setMatched();
                    break;
                }
            }
        }
    }
    return matchResult;
}

Matrix<float> MatchingManager::computeMatrix(vector<TrackedObject>& trackers,
                            const vector<Segmentation>& detections) {
    // Instantiate and populate Matrix
    Matrix<float> costMatrix(trackers.size(), detections.size());
    for (int i = 0, n = trackers.size(); i < n; ++i) {
        for (int j = 0, m = detections.size(); j < m; ++j) {
            Rect trackRect = trackers[i].bbox;
            Rect detectRect = toRect(detections[j]);

            if (config.distance == TrackerConfig::DistanceType::IOU) {
                costMatrix(i, j) = inverseIou(trackRect, detectRect);
            }
            else if (config.distance == TrackerConfig::DistanceType::EUCLIDEAN) {
                costMatrix(i, j) = euclidean(trackRect, detectRect);
            }
        }
    }
    return costMatrix;
}


// EARTH MOVERS DISTANCE

// Assign Matches
MatchingManager::MatchResult MatchingManager::matchEMD(vector<TrackedObject>& trackers,
                                                const vector<Segmentation>& detections,
                                                Mat& frame) {
    MatchResult matchResult = setMatchResult(trackers.size(), detections.size());
    if (trackers.size() == 0 || detections.size() == 0) return matchResult;

    // Calculate flow matrix
    Mat flow = computeFlow(trackers, detections, frame);

    // Convert flow (maximization) to cost matrix (minimization)
    int rows = flow.rows;
    int cols = flow.cols;
    Matrix<float> costMatrix(rows, cols);

    float maxFlow = 0.0f;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            maxFlow = std::max(maxFlow, flow.at<float>(i, j));

    // Fill cost matrix with (maxFlow - actualFlow) to minimize cost
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            costMatrix(i, j) = maxFlow - flow.at<float>(i, j);

    // Solve assignment
    Munkres<float> munkres;
    munkres.solve(costMatrix);

    // Parse results
    for (int i = 0, n = trackers.size(); i < n; ++i) {
        trackers[i].setUnmatched();
        for (int j = 0, m = detections.size(); j < m; ++j) {
            if (costMatrix(i, j) <= 0.03f && matchResult.unmatchedDetections.count(j)) {
                float iouResult = iou(trackers[i].bbox, toRect(detections[j]));
                if (iouResult >= 0.5f) {
                    matchResult.matches.emplace_back(i, j);
                    matchResult.unmatchedTrackers.erase(i);
                    matchResult.unmatchedDetections.erase(j);
                    trackers[i].setMatched();
                    break;
                }
            }
        }
    }

//    std::vector<std::tuple<int, int, float>> flowEntries;
//    // Flatten flow matrix to triplets
//    for (int i = 0; i < flow.rows; ++i) {
//        for (int j = 0; j < flow.cols; ++j) {
//            float f = flow.at<float>(i, j);
//            if (f > 0.0f) {
//                flowEntries.emplace_back(i, j, f);
//            }
//        }
//    }
//
//    // Sort by descending flow value
//    std::sort(flowEntries.begin(), flowEntries.end(), [](const auto& a, const auto& b)
//    { return std::get<2>(a) > std::get<2>(b); });
//
//    // Assign matches greedily in 1-to-1
//    for (const auto& [i, j, flowVal] : flowEntries) {
//        if (matchResult.unmatchedTrackers.count(i) && matchResult.unmatchedDetections.count(j)) {
//            // Threshold the match (no good otherwise)
//            float iou = distances::iou(trackers[i].bbox, toRect(detections[j]));
//            if (iou > 0.5f) {
//                matchResult.matches.emplace_back(i, j);
//                matchResult.unmatchedTrackers.erase(i);
//                matchResult.unmatchedDetections.erase(j);
//                trackers[i].setMatched();
//            }
//        }
//    }
    return matchResult;
}

// Calculate flow matrix
Mat MatchingManager::computeFlow(vector<TrackedObject>& trackers,
                const vector<Segmentation>& detections,
                Mat& frame) {
    // Get weights
    vector<float> trackWeights, detectWeights;
    if (config.emdWeight == TrackerConfig::EMDWeight::UNIFORM) {
        trackWeights = vector<float>(trackers.size(), 1.0f / trackers.size());
        detectWeights = vector<float>(detections.size(), 1.0f / detections.size());
    }
    else if (config.emdWeight == TrackerConfig::EMDWeight::CONFIDENCE) {
        trackWeights = collectWeights(trackers);
        detectWeights = collectWeights(detections);
    }

    // Get signatures
    cv::Mat sig1, sig2;
    if (config.emdSignature == TrackerConfig::EMDSignature::DISTANCE ||
        config.emdSignature == TrackerConfig::EMDSignature::AREA) {
        // Get center matrix
        vector<Rect> trackRects = collectRects(trackers);
        vector<Rect> detectRects = collectRects(detections);
        sig1 = computeSignature(trackRects, trackWeights);
        sig2 = computeSignature(detectRects, detectWeights);
    }
    else if (config.emdSignature == TrackerConfig::EMDSignature::AVERAGE_HOG) {
        std::vector<cv::Mat> trackerHOGs, detectHOGs;
        for (auto& t : trackers) {
            trackerHOGs.push_back(spatialPoolHOG(t.tracker.getTemplate()));
        }
        for (auto& d : detections) {
            // Instantiate a temp tracker of the detection to get HOG features
            KCFTracker tempTracker(config.HOG, config.FIXEDWINDOW, config.MULTISCALE, config.LAB);
            tempTracker.init(trackingUtils::toRect(d), frame);
            tempTracker.update(frame);
            detectHOGs.push_back(spatialPoolHOG(tempTracker.getTemplate()));
        }
        sig1 = computeHOGSignature(trackerHOGs, trackWeights);
        sig2 = computeHOGSignature(detectHOGs, detectWeights);
    }

    // Calculate flow
    Mat flow;
    float emd = EMD(sig1, sig2, cv::DIST_L2, noArray(), 0, flow);
    return flow;
}

template <typename T>
std::vector<float> MatchingManager::collectWeights(const vector<T>& objects) {
    vector<float> weights;
    // Collect weights
    for (const auto& obj : objects) {
        weights.push_back(obj.conf);
    }
    // Normalise weights
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    if (sum > 0) {
        for (auto& w : weights) w /= sum;
    }
    return weights;
}


vector<Rect> MatchingManager::collectRects(const vector<TrackedObject>& objects) {
    vector<Rect> rects;
    for (const auto& obj : objects)
        rects.push_back(obj.bbox);
    return rects;
}

vector<Rect> MatchingManager::collectRects(const vector<Segmentation>& objects) {
    vector<Rect> rects;
    for (const auto& obj : objects)
        rects.push_back(toRect(obj));
    return rects;
}


Mat MatchingManager::computeSignature(vector<Rect>& boxes, vector<float> weights) {
    // Instantiate signature
    size_t n = boxes.size();
    Mat signature;
    if (config.emdSignature == TrackerConfig::EMDSignature::DISTANCE) {
        signature = Mat(n, 3, CV_32F);
    }
    else {
        signature = Mat(n, 5, CV_32F);
    }
    // Populate signature
    for (size_t i = 0; i < n; ++i) {
        Point2d center = rectCenter(boxes[i]);
        signature.at<float>(i, 0) = weights[i];
        signature.at<float>(i, 1) = center.x;
        signature.at<float>(i, 2) = center.y;
        if (config.emdSignature == TrackerConfig::EMDSignature::AREA) {
            signature.at<float>(i, 3) = boxes[i].width;
            signature.at<float>(i, 4) = boxes[i].height;
        }
    }
    return signature;
}


cv::Mat MatchingManager::computeHOGSignature(const std::vector<cv::Mat>& hogDescriptors,
                                             const std::vector<float>& weights) {
    int N = static_cast<int>(hogDescriptors.size());
    int dims = hogDescriptors[0].rows * hogDescriptors[0].cols; // 31 * numCells

    cv::Mat signature(N, dims + 1, CV_32F);

    for (int i = 0; i < N; ++i) {
        signature.at<float>(i, 0) = weights[i];
        // Flatten HOG descriptor into a row vector
        cv::Mat flatHOG = hogDescriptors[i].reshape(1, 1); // 1 x (31*numCells)
        for (int d = 0; d < dims; ++d) {
            signature.at<float>(i, d + 1) = flatHOG.at<float>(0, d);
        }
    }
    return signature;
}


cv::Mat MatchingManager::spatialPoolHOG(const cv::Mat& tmpl) {
    int gridRows = 2;
    int gridCols = 2;
    int numFeatures = tmpl.rows; // 31
    int numCells = tmpl.cols;    // N
    int poolSize = gridRows * gridCols;
    int cellsPerRegion = numCells / poolSize;
    cv::Mat pooled(1, numFeatures * poolSize, CV_32F);

    int idx = 0;
    for (int gr = 0; gr < gridRows; ++gr) {
        for (int gc = 0; gc < gridCols; ++gc) {
            int start = (gr * gridCols + gc) * cellsPerRegion;
            int end = (gr * gridCols + gc + 1) * cellsPerRegion;
            end = std::min(end, numCells);
            if (start >= end) {
                pooled.colRange(idx, idx + numFeatures).setTo(0);
            } else {
                cv::Mat region = tmpl.colRange(start, end);
                cv::Mat avg;
                cv::reduce(region, avg, 1, cv::REDUCE_AVG);
                avg.reshape(1,1).copyTo(pooled.colRange(idx, idx + numFeatures));
            }
            idx += numFeatures;
        }
    }
    return pooled;
}


cv::Mat MatchingManager::computeAverageHOGSignature(const std::vector<cv::Mat>& hogDescriptors, const std::vector<float>& weights) {
    int N = static_cast<int>(hogDescriptors.size());
    int dims = 31;
    cv::Mat signature(N, dims + 1, CV_32F);
    for (int i = 0; i < N; ++i) {
        cv::Mat avgHOG;
        cv::reduce(hogDescriptors[i], avgHOG, 1, cv::REDUCE_AVG);  // avg over cols
        signature.at<float>(i, 0) = weights[i];  // weight or 1.0f if none
        for (int d = 0; d < dims; ++d) {
            signature.at<float>(i, d + 1) = avgHOG.at<float>(d, 0);
        }
    }
    return signature;
}

MatchingManager::MatchResult MatchingManager::setMatchResult(int trackerSize, int detectionSize) {
    MatchResult initMatchResult;
    for (int i = 0; i < trackerSize; ++i) {
        initMatchResult.unmatchedTrackers.insert(i);
    }
    for (int i = 0; i < detectionSize; ++i) {
        initMatchResult.unmatchedDetections.insert(i);
    }
    return initMatchResult;
}
