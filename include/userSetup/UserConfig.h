#pragma once

#include <string>

class UserConfig {

public:
    UserConfig();
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
        IOU,
        FEATUREMAPS
    };
    enum OutputType {
        SHOW,
        SAVE,
        BOTH,
        RESULTS
    };
    enum TestingType {
        NONE,
        MOT_ACCURACY,
        MOT_SWEEP
    };
    enum EMDWeight {
        UNIFORM,
        CONFIDENCE
    };
    enum OcclusionType {
        NO_OCCLUSION,
        RELAXED_REMOVAL,
        RELAXED_MATCHING,
        REMOVAL_AND_MATCHING
    };
    enum RemovalType {
        NO_REMOVAL,
        THRESHOLD,
        STRIKE_BASED
    };

    enum EMDSignature {
        DISTANCE,
        AREA,
        AVERAGE_HOG,
        Z_DIST_AREA
    };

    DistanceType distance;
    AssociationMethod association;
    TrackerType tracker;
    OutputType output;
    TestingType testing;
    EMDWeight emdWeight;
    EMDSignature emdSignature;
    OcclusionType occlusion;
    RemovalType removal;

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
    void setOcclusion(int occlusionSelected);
    void setRemoval(int removalSelected);
};
