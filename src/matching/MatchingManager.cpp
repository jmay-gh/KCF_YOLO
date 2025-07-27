#include "../include/matching/MatchingManager.h"

using namespace std;

MatchingManager::MatchingManager(const UserConfig& config, cv::Mat& frame)
        : config(config), currentFrame(frame), iouThreshold(0.5f) {

    using namespace DistanceCalculator;
    switch (config.distance) {
        case UserConfig::DistanceType::IOU:
            distanceFunc = [](const cv::Rect& a, const cv::Rect& b) { return inverseIou(a, b); };
            break;
        case UserConfig::DistanceType::EUCLIDEAN:
            distanceFunc = [](const cv::Rect& a, const cv::Rect& b) { return euclidean(a, b); };
            break;
        default:
            distanceFunc = [](const cv::Rect&, const cv::Rect&) { return 1.0f; };
    }
}


MatchingManager::MatchResult MatchingManager::setMatchResult(int trackerSize, int detectionSize) {
    MatchResult result;
    for (int i = 0; i < trackerSize; ++i) result.unmatchedTrackers.insert(i);
    for (int j = 0; j < detectionSize; ++j) result.unmatchedDetections.insert(j);
    return result;
}


MatchingManager::MatchResult MatchingManager::matchNN(vector<TrackedObject>& trackers,
                                                      const vector<Segmentation>& detections) {
    // Instantiate match results and return if empty
    MatchResult result = setMatchResult(trackers.size(), detections.size());
    if (trackers.empty() || detections.empty()) return result;

    // Create cost matrix and solve it
    auto costMatrix = CostMatrixBuilder::buildCostMatrix(trackers, detections, distanceFunc);
    auto matches = AssignmentSolver::solveGreedy(costMatrix);

    // Assign matches if they pass distance threshold
    for (const auto& [i, j] : matches) {
        float iouVal = DistanceCalculator::iou(trackers[i].bbox, toRect(detections[j]));
        if (iouVal > iouThreshold) {
            result.matches.emplace_back(i, j);
            result.unmatchedTrackers.erase(i);
            result.unmatchedDetections.erase(j);
        }
    }
    return result;
}

MatchingManager::MatchResult MatchingManager::matchHungarian(std::vector<TrackedObject>& trackers,
                                                             const std::vector<Segmentation>& detections) {
    // Instantiate match results and return if empty
    MatchResult result = setMatchResult(trackers.size(), detections.size());
    if (trackers.empty() || detections.empty()) return result;

    // Create cost matrix and solve it
    auto costMatrix = CostMatrixBuilder::buildCostMatrix(trackers, detections, distanceFunc);
    auto matches = AssignmentSolver::solveHungarian(costMatrix);

    // Assign matches if they pass distance threshold
    for (const auto& [i, j] : matches) {
        float iouVal = DistanceCalculator::iou(trackers[i].bbox, toRect(detections[j]));
        if (iouVal > iouThreshold) {
            result.matches.emplace_back(i, j);
            result.unmatchedTrackers.erase(i);
            result.unmatchedDetections.erase(j);
        }
    }
    return result;
}


MatchingManager::MatchResult MatchingManager::matchEMD(std::vector<TrackedObject>& trackers,
                                                       const std::vector<Segmentation>& detections) {
    using namespace DistanceCalculator;

    // Instantiate match results and return if empty
    MatchResult result = setMatchResult(trackers.size(), detections.size());
    if (trackers.empty() || detections.empty()) return result;

    // Create cost matrix and solve it
    auto costMatrix = CostMatrixBuilder::buildFlowMatrix(trackers, detections, currentFrame);
    auto matches = AssignmentSolver::solveHungarian(costMatrix);

    // Assign matches if they pass distance threshold
    for (auto& [i, j] : matches) {
        float iouVal = DistanceCalculator::iou(trackers[i].bbox, toRect(detections[j]));
        if (iouVal > 0.3) {
            result.matches.emplace_back(i, j);
            result.unmatchedTrackers.erase(i);
            result.unmatchedDetections.erase(j);
        }
    }
    return result;
}




// NEAREST NEIGHBOUR ALGO
//MatchingManager::MatchResult MatchingManager::matchNN(vector<TrackedObject>& trackers,
//                                                      const vector<Segmentation>& detections) {
//
//    MatchResult matchResult = setMatchResult(trackers.size(), detections.size());
//    if (trackers.size() == 0 || detections.size() == 0) return matchResult;
//
//    for (size_t i = 0; i < trackers.size(); ++i) {
//        int bestMatch = -1;
//        float bestDistance = numeric_limits<float>::max();
//        for (size_t j = 0; j < detections.size(); ++j) {
//
//            if (!matchResult.unmatchedDetections.count(j)) continue;
//
//            float distance;
//            if (config.distance == UserConfig::DistanceType::EUCLIDEAN) {
//                distance = euclidean(trackers[i].bbox, toRect(detections[j]));
//            }
//            else distance = inverseIou(trackers[i].bbox, toRect(detections[j]));
//            // Assign new match
//            if (distance < bestDistance) {
//                bestMatch = j;
//                bestDistance = distance;
//            }
//        }
//
//        if (bestMatch != -1) {
//            // Thresholding the match
//            float iouResult = iou(trackers[i].bbox, toRect(detections[bestMatch]));
//            if (iouResult > iouThreshold) {
//                matchResult.matches.emplace_back(i, bestMatch);
//                matchResult.unmatchedTrackers.erase(i);
//                matchResult.unmatchedDetections.erase(bestMatch);
//                trackers[i].setMatched();
//            }
//        }
//    }
//    return matchResult;
//}
//
//// HUNGARIAN ALGO
//MatchingManager::MatchResult MatchingManager::matchHungarian(vector<TrackedObject>& trackers,
//                                                             const vector<Segmentation>& detections) {
//
//    MatchResult matchResult = setMatchResult(trackers.size(), detections.size());
//    if (trackers.size() == 0 || detections.size() == 0) return matchResult;
//    // Create and solve cost matrix
//    Matrix<float> costMatrix;
//    costMatrix = computeMatrix(trackers, detections);
//    Munkres<float> munkres;
//    munkres.solve(costMatrix);
//
//    // Assign matches from cost matrix
//    for (int i = 0, n = trackers.size(); i < n; ++i) {
//        trackers[i].setUnmatched();
//        for (int j = 0, m = detections.size(); j < m; ++j) {
//            if (costMatrix(i, j) == 0.0f) {
//                float iouResult = iou(trackers[i].bbox, toRect(detections[j]));
//                if (iouResult > iouThreshold) {
//                    matchResult.matches.emplace_back(i, j);
//                    matchResult.unmatchedTrackers.erase(i);
//                    matchResult.unmatchedDetections.erase(j);
//                    trackers[i].setMatched();
//                    break;
//                }
//            }
//        }
//    }
//    return matchResult;
//}

//Matrix<float> MatchingManager::computeMatrix(vector<TrackedObject>& trackers,
//                                             const vector<Segmentation>& detections) {
//    // Instantiate cost matrix
//    Matrix<float> costMatrix(trackers.size(), detections.size());
//    // Compute cost matrix
//    for (int j = 0, n = detections.size(); j < n; ++j) {
//        KCFTracker dummyTracker(config.HOG, config.FIXEDWINDOW, config.MULTISCALE, config.LAB);
//        for (int i = 0, m = trackers.size(); i < m; ++i) {
//            if (config.distance == UserConfig::DistanceType::IOU) {
//                costMatrix(i, j) = inverseIou(trackers[i].bbox, toRect(detections[j]));
//            }
//            else if (config.distance == UserConfig::DistanceType::EUCLIDEAN) {
//                costMatrix(i, j) = euclidean(trackers[i].bbox, toRect(detections[j]));
//            }
//            else if (config.distance == UserConfig::DistanceType::FEATUREMAPS) {
//                Rect detectBox = resizeRect(toRect(detections[j]), trackers[i].bbox.width, trackers[i].bbox.height);
//                dummyTracker.init(detectBox, currentFrame);
//                costMatrix(i, j) = peakResponse(trackers[i], dummyTracker, toRect(detections[j]));
//            }
//        }
//    }
//    return costMatrix;
//}


// EARTH MOVERS DISTANCE

// Assign Matches
//MatchingManager::MatchResult MatchingManager::matchEMD(vector<TrackedObject>& trackers,
//                                                const vector<Segmentation>& detections,
//                                                Mat& frame) {
//
//    MatchResult matchResult = setMatchResult(trackers.size(), detections.size());
//    if (trackers.size() == 0 || detections.size() == 0) return matchResult;
//
//    // Calculate flow matrix
//    Mat flow = computeFlow(trackers, detections, frame);
//
//
//    std::vector<std::tuple<int, int, float>> flowEntries;
//    for (int i = 0; i < flow.rows; ++i) {
//        for (int j = 0; j < flow.cols; ++j) {
//            float f = flow.at<float>(i, j);
//            if (f > 0.0f) {
//                flowEntries.emplace_back(i, j, f);
//            }
//        }
//    }
//
////
////    // Convert flow (maximization) to cost matrix (minimization)
////    int rows = flow.rows;
////    int cols = flow.cols;
////    Matrix<float> costMatrix(rows, cols);
////
////    float maxFlow = 0.0f;
////    for (int i = 0; i < rows; ++i)
////        for (int j = 0; j < cols; ++j)
////            maxFlow = std::max(maxFlow, flow.at<float>(i, j));
////
////    // Fill cost matrix with (maxFlow - actualFlow) to minimize cost
////    for (int i = 0; i < rows; ++i)
////        for (int j = 0; j < cols; ++j)
////            costMatrix(i, j) = maxFlow - flow.at<float>(i, j);
////
////    // Solve assignment
////    Munkres<float> munkres;
////    munkres.solve(costMatrix);
////
////    // Parse results
////    for (int i = 0, n = trackers.size(); i < n; ++i) {
////        trackers[i].setUnmatched();
////        for (int j = 0, m = detections.size(); j < m; ++j) {
////            if (costMatrix(i, j) <= 0.03f && matchResult.unmatchedDetections.count(j)) {
////                float iouResult = iou(trackers[i].bbox, toRect(detections[j]));
////                if (iouResult >= 0.5f) {
////                    matchResult.matches.emplace_back(i, j);
////                    matchResult.unmatchedTrackers.erase(i);
////                    matchResult.unmatchedDetections.erase(j);
////                    trackers[i].setMatched();
////                    break;
////                }
////            }
////        }
////    }
//
////    std::map<std::pair<int, int>, float> objectFlow;
////    for (int i = 0; i < flow.rows; ++i) {
////        for (int j = 0; j < flow.cols; ++j) {
////            float f = flow.at<float>(i, j);
////            if (f > 0.0f) {
////                int track_id = outTrackLabels[i];
////                int det_id   = outDetectionLabels[j];
////                objectFlow[{track_id, det_id}] += f;
////            }
////        }
////    }
//
////    std::vector<std::pair<std::pair<int, int>, float>> sortedFlowEntries(
////            objectFlow.begin(), objectFlow.end()
////    );
////
////    // Sort by value in descending order
////    std::sort(sortedFlowEntries.begin(), sortedFlowEntries.end(),
////              [](const auto& a, const auto& b) {
////                  return a.second > b.second;
////              });
//
//    std::vector<std::tuple<int, int, float>> flowEntries;
//    for (int i = 0; i < flow.rows; ++i) {
//        for (int j = 0; j < flow.cols; ++j) {
//            float f = flow.at<float>(i, j);
//            if (f > 0.0f) {
//                flowEntries.emplace_back(i, j, f);
//            }
//        }
//    }
//
//    std::sort(flowEntries.begin(), flowEntries.end(), [](const auto& a, const auto& b) {
//        return std::get<2>(a) > std::get<2>(b);
//    });
//
//    for (const auto& [i, j, flowVal] : flowEntries) {
//        if (matchResult.unmatchedTrackers.count(i) && matchResult.unmatchedDetections.count(j)) {
//            float iouResult = iou(trackers[i].bbox, toRect(detections[j]));
//            if (iouResult > iouThreshold) {
//                matchResult.matches.emplace_back(i, j);
//                matchResult.unmatchedTrackers.erase(i);
//                matchResult.unmatchedDetections.erase(j);
//                trackers[i].setMatched();
//            }
//        }
//    }
//    return matchResult;
//}

// Calculate flow matrix
//Mat MatchingManager::computeFlow(vector<TrackedObject>& trackers,
//                                 const vector<Segmentation>& detections) {
//
//    // Get weights
//    vector<float> trackWeights, detectWeights;
//    if (config.emdWeight == UserConfig::EMDWeight::UNIFORM) {
//        trackWeights = vector<float>(trackers.size(), 1.0f / trackers.size());
//        detectWeights = vector<float>(detections.size(), 1.0f / detections.size());
//    }
//    else if (config.emdWeight == UserConfig::EMDWeight::CONFIDENCE) {
//        trackWeights = collectWeights(trackers);
//        detectWeights = collectWeights(detections);
//    }
//
//    // Get signatures
//    cv::Mat sig1, sig2;
//    if (config.emdSignature == UserConfig::EMDSignature::DISTANCE ||
//        config.emdSignature == UserConfig::EMDSignature::AREA ||
//        config.emdSignature == UserConfig::EMDSignature::Z_DIST_AREA) {
//        // Get center matrix
//        vector<pair<Rect, float>> trackRects = collectRects(trackers);
//        vector<pair<Rect, float>> detectRects = collectRects(detections);
//        sig1 = computeSignature(trackRects, trackWeights);
//        sig2 = computeSignature(detectRects, detectWeights);
//
//        // Calculate flow
//        Mat flow;
//        float emd = EMD(sig1, sig2, cv::DIST_L2, noArray(), 0, flow);
//        return flow;
//    }
//    else if (config.emdSignature == UserConfig::EMDSignature::AVERAGE_HOG) {
//        Mat costMatrix(trackers.size(), detections.size(), CV_32F);
//        std::vector<cv::Mat> trackerHOGs, detectHOGs;
//        vector<int> trackerHOGCols, detectHOGCols;
//        for (auto& t : trackers) {
//            trackerHOGs.push_back(spatialPoolHOG(t.tracker.getTemplate()));
//        }
//        for (auto& d : detections) {
//            // Instantiate a temp tracker of the detection to get HOG features
//            KCFTracker tempTracker(config.HOG, config.FIXEDWINDOW, config.MULTISCALE, config.LAB);
//            tempTracker.init(trackingUtils::toRect(d), frame);
//            detectHOGs.push_back(spatialPoolHOG(tempTracker.getTemplate()));
//        }
//        sig1 = computeHOGSignature(trackerHOGs, trackWeights);
//        sig2 = computeHOGSignature(detectHOGs, detectWeights);
//    }
//    else {
//        Mat costMatrix(trackers.size(), detections.size(), CV_32F);
//        std::vector<cv::Mat> trackerHOGs, detectHOGs;
//        vector<int> trackerHOGCols, detectHOGCols;
//
//        for (auto &t: trackers) {
////            trackerHOGs.push_back(spatialPoolHOG(t.tracker.getTemplate()));
//
//            trackerHOGs.push_back(t.tracker.getTemplate());
//            trackerHOGCols.push_back(t.tracker.tmplCols);
//        }
//        for (auto &d: detections) {
//            // Instantiate a temp tracker of the detection to get HOG features
//            KCFTracker tempTracker(config.HOG, config.FIXEDWINDOW, config.MULTISCALE, config.LAB);
//            tempTracker.init(trackingUtils::toRect(d), frame);
//
////            detectHOGs.push_back(spatialPoolHOG(tempTracker.getTemplate()));
//
//            detectHOGs.push_back(tempTracker.getTemplate());
//            detectHOGCols.push_back(tempTracker.tmplCols);
//        }
//        sig1 = newSignatureApproach(trackerHOGs, trackerHOGCols, outTrackLabels);
//        sig2 = newSignatureApproach(detectHOGs, detectHOGCols, outDetectionLabels);
//    }
//    // Calculate flow
//    Mat flow;
//    float emd = EMD(sig1, sig2, cv::DIST_L2, noArray(), 0, flow);
//    return flow;
//}
//
//template <typename T>
//
//std::vector<float> MatchingManager::collectWeights(const vector<T>& objects) {
//    vector<float> weights;
//    // Collect weights
//    for (const auto& obj : objects) {
//        weights.push_back(obj.conf);
//    }
//    // Normalise weights
//    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
//    if (sum > 0) {
//        for (auto& w : weights) w /= sum;
//    }
//    return weights;
//}
//
//
//vector<pair<Rect, float>> MatchingManager::collectRects(const vector<TrackedObject>& objects) {
//    vector<pair<Rect, float>> rects;
//    for (const auto& obj : objects)
//        rects.push_back({obj.bbox, obj.depth});
//    return rects;
//}
//
//vector<pair<Rect, float>> MatchingManager::collectRects(const vector<Segmentation>& objects) {
//    vector<pair<Rect, float>> rects;
//    for (const auto& obj : objects)
//        rects.push_back({toRect(obj), obj.depth});
//    return rects;
//}
//
//
//Mat MatchingManager::computeSignature(vector<pair<Rect, float>>& boxes, vector<float> weights) {
//    // Instantiate signature
//    size_t n = boxes.size();
//    Mat signature;
//    int cols = 3;
//    if (config.emdSignature == UserConfig::EMDSignature::AREA ||
//        config.emdSignature == UserConfig::EMDSignature::Z_DIST_AREA) {
//        cols += 2;
//    }
//    if (config.emdSignature == UserConfig::EMDSignature::Z_DIST_AREA) {
//        cols += 1;
//    }
//    signature = Mat(n, cols, CV_32F);
//
//    // Populate signature
//    for (size_t i = 0; i < n; ++i) {
//        Point2d center = rectCenter(boxes[i].first);
//        signature.at<float>(i, 0) = weights[i];
//        signature.at<float>(i, 1) = center.x;
//        signature.at<float>(i, 2) = center.y;
//        if (config.emdSignature == UserConfig::EMDSignature::AREA) {
//            signature.at<float>(i, 3) = boxes[i].first.width;
//            signature.at<float>(i, 4) = boxes[i].first.height;
//        }
//        if (config.emdSignature == UserConfig::EMDSignature::Z_DIST_AREA) {
//            // Assuming depth is stored in the tracker, otherwise modify accordingly
//            signature.at<float>(i, cols - 1) = boxes[i].second;
//        }
//    }
//    return signature;
//}
//
//cv::Mat MatchingManager::spatialPoolHOG(const cv::Mat& tmpl) {
//    int gridRows = 4;
//    int gridCols = 4;
//    int numFeatures = tmpl.rows;    // 31
//    int numCells = tmpl.cols;       // N
//    int poolSize = gridRows * gridCols;
//
//    int cellsPerRegion = numCells / poolSize;
//    cv::Mat pooled(1, numFeatures * poolSize, CV_32F);
//
//    int idx = 0;
//    for (int gr = 0; gr < gridRows; ++gr) {
//        for (int gc = 0; gc < gridCols; ++gc) {
//            int start = (gr * gridCols + gc) * cellsPerRegion;
//            int end = (gr * gridCols + gc + 1) * cellsPerRegion;
//            end = std::min(end, numCells);
//            if (start >= end) {
//                pooled.colRange(idx, idx + numFeatures).setTo(0);
//            } else {
//                cv::Mat region = tmpl.colRange(start, end);
//                cv::Mat avg;
//                cv::reduce(region, avg, 1, cv::REDUCE_AVG);
//                avg.reshape(1,1).copyTo(pooled.colRange(idx, idx + numFeatures));
//            }
//            idx += numFeatures;
//        }
//    }
//    return pooled;
//}


//cv::Mat MatchingManager::computeHOGSignature(const std::vector<cv::Mat>& hogDescriptors,
//                                             const std::vector<float>& weights) {
//    int N = static_cast<int>(hogDescriptors.size());
//    int dims = hogDescriptors[0].rows * hogDescriptors[0].cols; // 31 * numCells
//
//    cv::Mat signature(N, dims + 1, CV_32F);
//
//    for (int i = 0; i < N; ++i) {
//        signature.at<float>(i, 0) = weights[i];
//        // Flatten HOG descriptor into a row vector
//        cv::Mat flatHOG = hogDescriptors[i].reshape(1, 1); // 1 x (31*numCells)
//        for (int d = 0; d < dims; ++d) {
//            signature.at<float>(i, d + 1) = flatHOG.at<float>(0, d);
//        }
//    }
//    return signature;
//}


//cv::Mat MatchingManager::computeAverageHOGSignature(const std::vector<cv::Mat>& hogDescriptors, const std::vector<float>& weights) {
//    int N = static_cast<int>(hogDescriptors.size());
//    int dims = 31;
//    cv::Mat signature(N, dims + 1, CV_32F);
//    for (int i = 0; i < N; ++i) {
//        cv::Mat avgHOG;
//        cv::reduce(hogDescriptors[i], avgHOG, 1, cv::REDUCE_AVG);  // avg over cols
//        signature.at<float>(i, 0) = weights[i];  // weight or 1.0f if none
//        for (int d = 0; d < dims; ++d) {
//            signature.at<float>(i, d + 1) = avgHOG.at<float>(d, 0);
//        }
//    }
//    return signature;
//}

//
//Mat MatchingManager::newSignatureApproach(const std::vector<cv::Mat>& hogDescriptors, vector<int> hogSizes, vector<int>& objectIds) {
//
//    long long size = 0;
//    for (int i = 0; i < hogDescriptors.size(); ++i) {
//        cout << hogDescriptors[i].cols << endl;
//        size += (hogDescriptors[i].cols * 9);
//    }
//
//    Mat signature(size, 4, CV_32F);
//    // Iterate all HOGS
//    int totalRow = 0;
//    for (int i = 0; i < hogDescriptors.size(); ++i) {
//        // Iterate the rows
//
//        for (int j = 0; j < hogDescriptors[i].cols; ++j) {
//            // Iterate the orientations
//            int x = j % hogSizes[i];
//            int y = j / hogSizes[i];
//
//            float weight = 0;
//            float angle = 0;
//            for (int k = 0; k < 9; ++k) {
//                int baseIndex = 31 - 9; // last 9 bins start here
//                float currWeight = hogDescriptors[i].at<float>(baseIndex + k, j);
//
//                if (currWeight > weight) {
//                    weight = currWeight;
//                    angle = static_cast<float>(k) * 20.0f;
//                }
//            }
//            if (weight < 0) cout << weight << endl;
//
//            signature.at<float>(totalRow, 0) = weight;
//            signature.at<float>(totalRow, 1) = x;
//            signature.at<float>(totalRow, 2) = y;
//            signature.at<float>(totalRow, 3) = angle;
//            totalRow++;
//            objectIds.push_back(i);
//        }
//    }
//    return signature;
//}


//
//Mat MatchingManager::resizeMap(const Mat& input) {
//    Mat output(6, 6, CV_32F);
//
//    // Create complex input
//    Mat complexInput;
//    cv::dft(input, complexInput, cv::DFT_COMPLEX_OUTPUT);
//
//    // Shift input
//    Mat shiftedInput;
//    cv::dft(complexInput, shiftedInput, cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);
//    Mat fftShifted;
//    cv::dft(shiftedInput, fftShifted, cv::DFT_COMPLEX_OUTPUT);
//
//    // Center input
//    Mat centeredInput = fftshift(fftShifted);
//
//    // Crop or pad depending on size
//    cv::Size outputSize(6, 6);
//    Mat resizedInput = fourierCropOrPad(centeredInput, outputSize);
//
//    // Reshift the output -- maybe not needed as symmetric?
//    cv::Mat reshiftInput = fftshift(resizedInput);
//
//    cv::Mat inverse;
//    cv::dft(reshiftInput, inverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
//
//    // Gain adjustment
//    float gain = (float)(outputSize.width * outputSize.height) / (input.cols * input.rows);
//    inverse *= gain;
//
//    return inverse;
//}

//
//cv::Mat MatchingManager::fourierCropOrPad(const cv::Mat& input, cv::Size targetSize) {
//    cv::Mat output = cv::Mat::zeros(targetSize, input.type());
//
//    int minRows = std::min(input.rows, targetSize.height);
//    int minCols = std::min(input.cols, targetSize.width);
//
//    int yOffsetIn = (input.rows - minRows) / 2;
//    int xOffsetIn = (input.cols - minCols) / 2;
//
//    int yOffsetOut = (targetSize.height - minRows) / 2;
//    int xOffsetOut = (targetSize.width - minCols) / 2;
//
//    input(cv::Rect(xOffsetIn, yOffsetIn, minCols, minRows))
//            .copyTo(output(cv::Rect(xOffsetOut, yOffsetOut, minCols, minRows)));
//
//    return output;
//}
