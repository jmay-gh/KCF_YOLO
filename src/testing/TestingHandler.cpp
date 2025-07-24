#include "../../include/testing/TestingHandler.h"

using namespace testing;
using namespace std;

TestingHandler::TestingHandler(TrackerConfig& config) : config(config) { }

void TestingHandler::runTests(const string& gtFilePath, const string& framesFilePath) {
    // Run tests implementation
    std::cout << "Running tests..." << std::endl;

    // Add test cases here
    if (config.testing == TrackerConfig::TestingType::MOT_ACCURACY) {
        runMOTAccuracyTests(gtFilePath, framesFilePath);
    } else if (config.testing == TrackerConfig::TestingType::MOT_SWEEP) {
        runMOTAccuracySweep(gtFilePath, framesFilePath);
    } else {
        std::cout << "No tests to run." << std::endl;
    }
}

void TestingHandler::runMOTAccuracyTests(const string& gtFilePath, const string& framesFilePath) {
    // Example test case for MOT accuracy
    cout << "Running MOT accuracy tests..." << endl;

    map<int, vector<GroundTruth>> groundTruths;
    testing::parseGroundTruth(gtFilePath, groundTruths);
    testing::calculateMOTAccuracy(groundTruths, framesFilePath, config);
}

void TestingHandler::runMOTAccuracySweep(const string& gtFilePath, const string& framesFilePath) {
    // Example test case for MOT accuracy
    cout << "Running MOT accuracy sweep..." << endl;
    map<int, vector<GroundTruth>> groundTruths;
    testing::parseGroundTruth(gtFilePath, groundTruths);
    testing::calculateMOTAccuracyRange(groundTruths, framesFilePath, config);
}



