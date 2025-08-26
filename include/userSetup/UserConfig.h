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
        CM_EMD,
        FM_EMD
    };
    enum DistanceType {
        EUCLIDEAN,
        IOU
    };
    enum AuxType {
        NONE,
        AREA,
        HOG_FEATURES,
        DEPTH,
        VELOCITY
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
    enum OutputType {
        SHOW,
        SAVE,
        BOTH,
        RESULTS
    };
//    enum EMDSignature {
//        DISTANCE,
//        AREA,
//        AVERAGE_HOG,
//        Z_DIST_AREA
//    };

    DistanceType distance;
    AssociationMethod association;
    TrackerType tracker;
    OutputType output;
    EMDWeight emdWeight;

    AuxType auxType;

//    EMDSignature emdSignature;
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
    void setAux(int auxSelected);
    void setOutput(int outputSelected);
    void setEMD(int emdWeightSelected);
    void setOcclusion(int occlusionSelected);
    void setRemoval(int removalSelected);
};
