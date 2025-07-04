#pragma once

#include "state.hpp"
#include "heatmap.hpp"
#include "linechart.hpp"
#include "nn.hpp"
#include "dataset.hpp"
#include <map>
#include <string>
#include <algorithm>

// Helper function to find a key from a value in a map
template<typename K, typename V>
std::string getKeyFromValue(const std::map<K, V>& map, const V& value) {
    auto it = std::find_if(map.begin(), map.end(),
                           [&value](const auto& pair) {
                               return pair.second == value;
                           });
    if (it != map.end()) {
        return it->first;
    }
    return ""; // Return an empty string or handle as an error if not found
}


// Helper function
inline double map_range(double val, double in_min, double in_max, double out_min, double out_max) {
    if (in_max == in_min) {
        return out_min;
    }
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

class PlaygroundApp {
public:
    PlaygroundApp();
    ~PlaygroundApp();

    void runFrame();
    void drawUI();

private:
    void reset(bool onStartup = false);
    void oneStep();
    void generateData(bool firstTime = false);
    void updateUIState();

    void drawControls();
    void drawNetwork();
    void drawOutput();

    std::vector<std::string> constructInputIds();
    std::vector<double> constructInput(double x, double y);
    void updateDecisionBoundary();
    double getLoss(nn::Network& net, const std::vector<playground::Example2D>& data);

    State state;
    nn::Network network;

    std::vector<playground::Example2D> trainData;
    std::vector<playground::Example2D> testData;

    static const int DENSITY = 50;
    const std::pair<double, double> xDomain = {-6.0, 6.0};
    const std::pair<double, double> yDomain = {-6.0, 6.0};

    HeatMap mainHeatMap;
    std::map<std::string, HeatMap> nodeHeatMaps;
    LineChart lineChart;

    bool isPlaying = false;
    bool parametersChanged = false;
    int iter = 0;
    double lossTrain = 0;
    double lossTest = 0;

    std::map<std::string, ImVec2> node2coord;
    std::string selectedNodeId;

    std::map<std::string, std::vector<std::vector<double>>> boundary;
};
