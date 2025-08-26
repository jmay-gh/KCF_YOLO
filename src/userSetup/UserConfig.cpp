#include "userSetup/UserConfig.h"
#include <iostream>


UserConfig::UserConfig() { }

void UserConfig::setAssociation(int associationSelected) {
    // Set the association method
    association = static_cast<AssociationMethod>(associationSelected);
}

void UserConfig::setEMD(int emdWeightSelected) {
    emdWeight = static_cast<EMDWeight>(emdWeightSelected);
//    emdSignature = static_cast<EMDSignature>(emdSignatureSelected);
}

void UserConfig::setTracker(int trackerSelected) {
    // Set the tracker type
    tracker = static_cast<TrackerType>(trackerSelected);
}

void UserConfig::setAux(int auxSelected) {
    std::cout << "AUX SET" << std::endl;
    auxType = static_cast<AuxType>(auxSelected);
}

void UserConfig::setKCFTracker(int kcftrackerSelected) {
    if (kcftrackerSelected == 0) { HOG = false; LAB = false; }
    else if (kcftrackerSelected == 1) { HOG = true; LAB = false; }
    else if (kcftrackerSelected == 2) { HOG = true; LAB = true; }
    MULTISCALE = true;
    FIXEDWINDOW = true;
}

void UserConfig::setDistance(int distanceSelected) {
    // Set the distance metric
    distance = static_cast<DistanceType>(distanceSelected);
}

void UserConfig::setOcclusion(int occlusionSelected) {
    // Set the occlusion handling type
    occlusion = static_cast<OcclusionType>(occlusionSelected);
}

void UserConfig::setRemoval(int removalSelected) {
    // Set the removal type
    removal = static_cast<RemovalType>(removalSelected);
}

void UserConfig::setOutput(int outputSelected) {
    // Set the output type
    output = static_cast<OutputType>(outputSelected);
}