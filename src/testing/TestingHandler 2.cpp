#include "../../include/testing/TestingHandler.h"

using namespace testing;

TestingHandler::TestingHandler(TrackerConfig& config) : config(config) { }

void TestingHandler::runTests(const std::string& filePath, Mat& frame) {
    // Run tests implementation
    std::cout << "Running tests..." << std::endl;

    // Add test cases here
    if (config.testing == TrackerConfig::TestingType::MOT_ACCURACY) {
        runMOTAccuracyTests("../img/deer_7/ground_truth.txt", frame);
    } else {
        std::cout << "No tests to run." << std::endl;
    }
}

void TestingHandler::runMOTAccuracyTests(const std::string& filePath, Mat& frame) {
    // Example test case for MOT accuracy
    std::cout << "Running MOT accuracy tests..." << std::endl;
    // Add test cases here
    std::map<int, std::vector<GroundTruth>> groundTruths;
    testing::parseGroundTruth(filePath, groundTruths);
    testing::calculateMOTAccuracy(groundTruths, config, frame);
}



