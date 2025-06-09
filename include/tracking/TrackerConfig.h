#pragma once

#include <string>

class TrackerConfig {

public:
    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool SILENT = true;
    bool LAB = false;

    static TrackerConfig parseArgs(int argc, char** argv);
};
