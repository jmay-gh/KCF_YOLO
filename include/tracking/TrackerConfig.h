#pragma once

#include <string>

class TrackerConfig {

public:
    TrackerConfig();
    enum TrackerType {
        COTRACKER,
        KERNELISED_CORRELATION_FILTER
    };
    enum AssociationMethod {
        NEAREST_NEIGHBOUR,
        HUNGARIAN_ALGORITHM,
        GROUND_MOVERS_DISTANCE
    };
    enum DistanceType {
        EUCLIDEAN,
        COSINE,
        IOU
    };
    enum OutputType {
        SHOW,
        SAVE,
        BOTH
    };
    enum TestingType {
        NONE,
        MOT_ACCURACY,
    };
    enum EMDWeight {
        UNIFORM,
        CONFIDENCE
    };
    enum EMDSignature {
        DISTANCE,
        AREA,
        AVERAGE_HOG
    };

    DistanceType distance;
    AssociationMethod association;
    TrackerType tracker;
    OutputType output;
    TestingType testing;
    EMDWeight emdWeight;
    EMDSignature emdSignature;

    bool HOG;
    bool FIXEDWINDOW;
    bool MULTISCALE;
    bool SILENT;
    bool LAB;

    void setTracker(int trackerSelected);
    void setKCFTracker(int kcftrackerSelected);
    void setAssociation(int associationSelected);
    void setDistance(int distanceSelected);
    void setOutput(int outputSelected);
    void setTesting(int testingSelected);
    void setEMD(int emdWeightSelected, int emdSignatureSelected);
};
