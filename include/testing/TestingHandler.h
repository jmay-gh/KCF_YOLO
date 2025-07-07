#pragma once

#include "../tracking/TrackerConfig.h"
#include "MOTAccuracy.hpp"
#include "../matching/MatchingManager.h"

using namespace std;

class TestingHandler {

    public:
        // Constructor
        TestingHandler(TrackerConfig& config);
        TrackerConfig& config;
        // Run tests
        void runTests(const string& filePath, cv::Mat& frame);
        void runMOTAccuracyTests(const string& filePath, cv::Mat& frame);
};

