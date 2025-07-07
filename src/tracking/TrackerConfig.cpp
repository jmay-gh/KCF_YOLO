#include "../../include/tracking/TrackerConfig.h"

TrackerConfig::TrackerConfig() { }

void TrackerConfig::setAssociation(int associationSelected) {
    // Set the association method
    association = static_cast<AssociationMethod>(associationSelected);
}

void TrackerConfig::setEMD(int emdWeightSelected, int emdSignatureSelected) {
    emdWeight = static_cast<EMDWeight>(emdWeightSelected);
    emdSignature = static_cast<EMDSignature>(emdSignatureSelected);
}

void TrackerConfig::setTracker(int trackerSelected) {
    // Set the tracker type
    tracker = static_cast<TrackerType>(trackerSelected);
}

void TrackerConfig::setKCFTracker(int kcftrackerSelected) {
    if (kcftrackerSelected == 0) { HOG = false; LAB = false; }
    else if (kcftrackerSelected == 1) { HOG = true; LAB = false; }
    else if (kcftrackerSelected == 2) { HOG = true; LAB = true; }
    MULTISCALE = true;
    FIXEDWINDOW = true;
}

void TrackerConfig::setDistance(int distanceSelected) {
    // Set the distance metric
    distance = static_cast<DistanceType>(distanceSelected);
}

void TrackerConfig::setOutput(int outputSelected) {
    // Set the output type
    output = static_cast<OutputType>(outputSelected);
}

void TrackerConfig::setTesting(int testingSelected) {
    // Set the testing type
    testing = static_cast<TestingType>(testingSelected);
}