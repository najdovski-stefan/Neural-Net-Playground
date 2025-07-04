#pragma once

#include "dataset.hpp"
#include "nn.hpp"
#include <string>
#include <vector>
#include <map>

// Forward declaration
struct State;

// Maps for converting strings to function pointers
extern std::map<std::string, nn::ActivationFunction> activations;
extern std::map<std::string, const nn::RegularizationFunction*> regularizations;
// Use the correct 'playground' namespace
extern std::map<std::string, playground::DataGenerator> datasets;
extern std::map<std::string, playground::DataGenerator> regDatasets;

enum class Problem { CLASSIFICATION, REGRESSION };
extern std::map<std::string, Problem> problems;

// Helper to get a key from a map by its value.
template<typename M, typename V>
std::string getKeyFromValue(const M& map, const V& value) {
    for (const auto& pair : map) {
        if constexpr (std::is_pointer_v<V>) {
            if (pair.second == value) {
                return pair.first;
            }
        }
    }
    return "";
}


struct State {
    float learningRate = 0.03f;
    float regularizationRate = 0.0f;
    float noise = 0.0f;

    bool showTestData = false;
    bool showDataPoints = true;
    bool showOverfit = false;
    int batchSize = 10;
    bool discretize = false;
    int percTrainData = 70;

    std::string activationKey = "tanh";
    const nn::RegularizationFunction* regularization = nullptr;
    Problem problem = Problem::CLASSIFICATION;

    bool initZero = false;
    bool collectStats = false;j

    int numHiddenLayers = 1;
    std::vector<int> networkShape = {4, 2};

    // Feature flags
    bool x = true;
    bool y = true;
    bool xTimesY = false;
    bool xSquared = false;
    bool ySquared = false;
    bool cosX = false;
    bool sinX = false;

    int numSamples = 500;
    playground::DataGenerator dataset = playground::classifyCircleData;
    playground::DataGenerator regDataset = playground::regressPlane;
    std::string seed;

    void resetToDefaults() {
        learningRate = 0.03f;
        regularizationRate = 0.0f;
        noise = 0.0f;
        showTestData = false;
        showOverfit = false;
        batchSize = 10;
        discretize = false;
        percTrainData = 50;
        activationKey = "tanh";
        regularization = nullptr;
        problem = Problem::CLASSIFICATION;
        initZero = false;
        numHiddenLayers = 1;
        networkShape = {4, 2};
        x = true;
        y = true;
        xTimesY = false;
        xSquared = false;
        ySquared = false;
        cosX = false;
        sinX = false;
        dataset = playground::classifyCircleData;
        regDataset = playground::regressPlane;
    }
};
