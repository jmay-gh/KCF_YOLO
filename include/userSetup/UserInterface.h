#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include "UserConfig.h"
#include <iostream>

class UserInterface {

public:
    UserInterface();
    void run(UserConfig& config);
//    void outputSettings(UserConfig& config);
};