#include "userSetup/UserInterface.h"

using namespace ftxui;

UserInterface::UserInterface() { }


void UserInterface::run(UserConfig& config) {
    // --- State Variables ---
    int trackerSelected = 1;
    int tab_selected = 0;
    int kcftrackerSelected = 1;
    int associationSelected = 2;
    int distanceSelected = 2;
    int outputSelected = 0;
    int emdWeightsSelected = 0;
    int emdSigSelected = 0;
    int occlusionSelected = 0;
    int removalSelected = 0;

    // --- Labels ---
    std::vector<std::string> trackerOptions = {"CoTracker", "KCF Tracker"};

    std::vector<std::string> tab_labels = {"General", "Association", "Distance", "Occlusions", "Removals", "Output"};

    std::vector<std::string> kcftrackerOptions = {"Grayscale", "HOG", "HOG + LAB"};
    std::vector<std::string> associationOptions = {"Nearest Neighbour", "Hungarian Algorithm", "Ground Movers Distance"};
    std::vector<std::string> distanceOptions = {"Euclidean Distance", "Cosine", "Intersection over Union", "HOG"};

    std::vector<std::string> removalOptions = {"None", "Threshold Removal", "Strike Based Removal"};
    std::vector<std::string> occlusionOptions = {"None", "Relaxed Removal", "Relaxed Matching", "Both"};

    std::vector<std::string> outputOptions = {"Play live", "Save to file", "Both", "Results"};
    std::vector<std::string> emdWeights = {"Equal weighting", "Confidence weighting"};
    std::vector<std::string> emdSig = {"Distance", "Distance + Area", "HOG features", "3D Distance + Area"};


    // --- Components ---
    auto trackerSelector = Radiobox(&trackerOptions, &trackerSelected);
    auto tabToggle = Toggle(&tab_labels, &tab_selected);

    auto kcftrackerDropdown = Radiobox(&kcftrackerOptions, &kcftrackerSelected);
    auto associationDropdown = Radiobox(&associationOptions, &associationSelected);
    auto distanceDropdown = Radiobox(&distanceOptions, &distanceSelected);

    auto occlusionDropdown = Radiobox(&occlusionOptions, &occlusionSelected);
    auto removalDropdown = Radiobox(&removalOptions, &removalSelected);

    auto outputDropdown = Radiobox(&outputOptions, &outputSelected);
    auto emdWeightsDropdown = Radiobox(&emdWeights, &emdWeightsSelected);
    auto emdSigDropdown = Radiobox(&emdSig, &emdSigSelected);

    // --- Tab Containers ---
    Component cotrackerTabs = Container::Tab({
//        Renderer([] { return text("CoTracker: General settings (none)"); }),
//        Renderer([] { return text("Not applicable for Association"); }),
//        Renderer([] { return text("Not applicable for Distance"); }),
        outputDropdown
    }, &tab_selected);

    Component kcfTabs = Container::Tab({
        kcftrackerDropdown,
        associationDropdown,
        distanceDropdown,
        occlusionDropdown,
        removalDropdown,
        outputDropdown
    }, &tab_selected);

    // --- Dynamic Renderer for Active Tabs ---
    Component tabContainer = Container::Tab({
        cotrackerTabs,
        kcfTabs
    }, &trackerSelected);

    // --- Screen ---
    auto screen = ScreenInteractive::Fullscreen();

    // --- Confirm Button ---
    auto confirmButton = Button("Confirm", [&] {
        screen.ExitLoopClosure()();
    });

    // --- Root Component ---
    Component rootContainer = Container::Vertical({
        trackerSelector,
        tabToggle,
        tabContainer,
        emdWeightsDropdown,
        emdSigDropdown,
        confirmButton
    });

    // --- Layout Renderer ---
    auto renderer = Renderer(rootContainer, [&] {
        std::vector<Element> right_pane;
        right_pane.push_back(tabToggle->Render());
        right_pane.push_back(separator());
        right_pane.push_back(tabContainer->Render());
        // Add in EMD settings if selected
        if (trackerSelected == 1 && tab_selected == 1 && associationSelected == 2) {
            right_pane.push_back(separator());
            right_pane.push_back(text("EMD Weightings:"));
            right_pane.push_back(emdWeightsDropdown->Render());
            right_pane.push_back(separator());
            right_pane.push_back(text("EMD Signatures:"));
            right_pane.push_back(emdSigDropdown->Render());
        }

        auto layout = vbox({
            hbox({
                window(text("Tracker Type:"), trackerSelector->Render())
                | size(WIDTH, EQUAL, 30),
                separator(),
                vbox(right_pane) | flex
            }) | flex,
            separator(),
            confirmButton->Render() | center
        }) | border;

        return layout;
    });

    // --- Loop ---
    screen.Loop(renderer);

    // --- Save to Config ---
    config.setTracker(trackerSelected);
    config.setKCFTracker(kcftrackerSelected);
    config.setAssociation(associationSelected);
    config.setEMD(emdWeightsSelected, emdSigSelected);
    config.setDistance(distanceSelected);
    config.setOcclusion(occlusionSelected);
    config.setRemoval(removalSelected);
    config.setOutput(outputSelected);

    std::cout << "Tracker Type: " << trackerOptions[config.tracker] << std::endl;
    std::cout << "KCF Tracker Type: " << kcftrackerOptions[config.HOG] << std::endl;
    std::cout << "Association Method: " << associationOptions[config.association] << std::endl;
    std::cout << "Distance Metric: " << distanceOptions[config.distance] << std::endl;
    std::cout << "Output Type: " << outputOptions[config.output] << std::endl;
    std::cout << "EMD Weighting: " << emdWeights[config.emdWeight] << std::endl;
    std::cout << "EMD Signature: " << emdSig[config.emdSignature] << std::endl;
    std::cout << "Occlusion Handling: " << occlusionOptions[config.occlusion] << std::endl;
    std::cout << "Removal Type: " << removalOptions[config.removal] << std::endl;
    std::cout << "Output Type: " << outputOptions[config.output] << std::endl;
}


