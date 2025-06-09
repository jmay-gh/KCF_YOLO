#include "tracking/TrackerConfig.h"

TrackerConfig TrackerConfig::parseArgs(int argc, char** argv) {
    TrackerConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "hog") config.HOG = true;
        else if (arg == "fixed_window") config.FIXEDWINDOW = true;
        else if (arg == "singlescale") config.MULTISCALE = false;
        else if (arg == "show") config.SILENT = false;
        else if (arg == "lab") { config.LAB = true; config.HOG = true; }
        else if (arg == "gray") config.HOG = false;
    }
    return config;
}
