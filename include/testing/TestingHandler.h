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
        void runTests(const string& gtFilePath, const string& framesFilePath);
        void runMOTAccuracyTests(const string& gtFilePath, const string& framesFilePath);
        void runMOTAccuracySweep(const string& gtFilePath, const string& framesFilePath);

};

