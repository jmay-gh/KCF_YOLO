#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include "tracking/TrackerConfig.h"
#include <iostream>

class UserInterface {

public:
    UserInterface();
    void run(TrackerConfig& config);
};