#include "playground.hpp"
#include <imgui.h>
#include <implot.h>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>


// Initialize static maps from state.hpp
std::map<std::string, nn::ActivationFunction> activations = {
    {"relu", nn::Activations::RELU},
    {"tanh", nn::Activations::TANH},
    {"sigmoid", nn::Activations::SIGMOID},
    {"linear", nn::Activations::LINEAR}
};
std::map<std::string, const nn::RegularizationFunction*> regularizations = {
    {"none", nullptr},
    {"L1", &nn::RegularizationFunctions::L1},
    {"L2", &nn::RegularizationFunctions::L2}
};
std::map<std::string, playground::DataGenerator> datasets = {
    {"circle", playground::classifyCircleData},
    {"xor", playground::classifyXORData},
    {"gauss", playground::classifyTwoGaussData},
    {"spiral", playground::classifySpiralData},
    {"star", playground::classifyStarData},
    {"sine", playground::classifySineData},
    {"checkerboard", playground::classifyCheckerboardData},
    {"moons", playground::classifyMoonsData},
    {"heart", playground::classifyHeartData},
};
std::map<std::string, playground::DataGenerator> regDatasets = {
    {"reg-plane", playground::regressPlane},
    {"reg-gauss", playground::regressGaussian}
};
std::map<std::string, Problem> problems = {
    {"classification", Problem::CLASSIFICATION},
    {"regression", Problem::REGRESSION}
};

// --- Feature Definitions ---
struct InputFeature {
    std::function<double(double, double)> f;
    std::string label;
};
std::map<std::string, InputFeature> INPUTS = {
    {"x", {[](double x, double y) { return x; }, "X_1"}},
    {"y", {[](double x, double y) { return y; }, "X_2"}},
    {"xSquared", {[](double x, double y) { return x * x; }, "X_1^2"}},
    {"ySquared", {[](double x, double y) { return y * y; }, "X_2^2"}},
    {"xTimesY", {[](double x, double y) { return x * y; }, "X_1X_2"}},
    {"sinX", {[](double x, double y) { return std::sin(x); }, "sin(X_1)"}},
};

// --- PlaygroundApp Implementation ---

PlaygroundApp::PlaygroundApp() : mainHeatMap(DENSITY, xDomain, yDomain) {
    reset(true);
}

PlaygroundApp::~PlaygroundApp() {
    if (!network.empty()) {
        nn::deleteNetwork(network);
    }
}


void PlaygroundApp::runFrame() {
    if (isPlaying) {
        oneStep();
    }
}

void PlaygroundApp::drawUI() {
    ImGui::DockSpace(ImGui::GetID("MyDockSpace"));

    if (ImGui::Begin("Controls")) {
        drawControls();
    }
    ImGui::End();

    if (ImGui::Begin("Network")) {
        drawNetwork();
    }
    ImGui::End();

    if (ImGui::Begin("Output")) {
        drawOutput();
    }
    ImGui::End();
}



void PlaygroundApp::drawControls() {
    // --- Top Toolbar ---
    if (ImGui::Button(isPlaying ? "Pause" : "Play")) { isPlaying = !isPlaying; }
    ImGui::SameLine();
    if (ImGui::Button("Step")) { oneStep(); }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) { reset(); }
    ImGui::SameLine();
    ImGui::Text("Epoch: %s", std::to_string(iter).c_str());

    ImGui::Separator();

    // --- Data Section ---
    if (ImGui::CollapsingHeader("Data", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* items[] = { "Classification", "Regression" };
        int current_problem = static_cast<int>(state.problem);
        if (ImGui::Combo("Problem type", &current_problem, items, IM_ARRAYSIZE(items))) {
            state.problem = static_cast<Problem>(current_problem);
            parametersChanged = true;
            reset();
        }

        if (state.problem == Problem::CLASSIFICATION) {
            ImGui::Text("Dataset:"); ImGui::SameLine();
            if (ImGui::Button("Circle")) { state.dataset = playground::classifyCircleData; parametersChanged = true; reset(); } ImGui::SameLine();
            if (ImGui::Button("XOR")) { state.dataset = playground::classifyXORData; parametersChanged = true; reset(); } ImGui::SameLine();
            if (ImGui::Button("Gauss")) { state.dataset = playground::classifyTwoGaussData; parametersChanged = true; reset(); } ImGui::SameLine();
            if (ImGui::Button("Spiral")) { state.dataset = playground::classifySpiralData; parametersChanged = true; reset(); } ImGui::SameLine();
            if (ImGui::Button("Star")) { state.dataset = playground::classifyStarData; parametersChanged = true; reset(); } ImGui::SameLine();
            if (ImGui::Button("Sine")) { state.dataset = playground::classifySineData; parametersChanged = true; reset(); } ImGui::SameLine();
            if (ImGui::Button("Checkerboard")) { state.dataset = playground::classifyCheckerboardData; parametersChanged = true; reset(); } ImGui::SameLine();
            if (ImGui::Button("Moons")) { state.dataset = playground::classifyMoonsData; parametersChanged = true; reset(); } ImGui::SameLine();
            if (ImGui::Button("Heart")) { state.dataset = playground::classifyHeartData; parametersChanged = true; reset(); }
        } else {
            ImGui::Text("Dataset:"); ImGui::SameLine();
            if (ImGui::Button("Plane")) { state.regDataset = playground::regressPlane; parametersChanged = true; reset(); } ImGui::SameLine();
            if (ImGui::Button("Gauss")) { state.regDataset = playground::regressGaussian; parametersChanged = true; reset(); }
        }

        ImGui::SliderInt("Ratio of training data", &state.percTrainData, 10, 90, "%d%%");
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            parametersChanged = true;
            reset();
        }
        ImGui::SliderFloat("Noise", &state.noise, 0.0f, 0.5f, "%.2f");
        ImGui::SliderInt("Batch size", &state.batchSize, 1, 30);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            parametersChanged = true;
            reset();
        }
        ImGui::SliderInt("Number of samples", &state.numSamples, 100, 2000);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            parametersChanged = true;
            reset();
        }
    }

    ImGui::Separator();

    // --- Features Section ---
    if (ImGui::CollapsingHeader("Features", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Which features to use:");
        if (ImGui::Checkbox("X1", &state.x)) { parametersChanged = true; reset(); }
        if (ImGui::Checkbox("X2", &state.y)) { parametersChanged = true; reset(); }
        if (ImGui::Checkbox("X1^2", &state.xSquared)) { parametersChanged = true; reset(); }
        if (ImGui::Checkbox("X2^2", &state.ySquared)) { parametersChanged = true; reset(); }
        if (ImGui::Checkbox("X1*X2", &state.xTimesY)) { parametersChanged = true; reset(); }
        if (ImGui::Checkbox("sin(X1)", &state.sinX)) { parametersChanged = true; reset(); }
    }

    ImGui::Separator();

    // --- Hidden Layers Section ---
    if (ImGui::CollapsingHeader("Hidden Layers", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Number of hidden layers");
        ImGui::SameLine();
        if (ImGui::Button("+") && state.numHiddenLayers < 6) {
            state.numHiddenLayers++;
            state.networkShape.push_back(2);
            parametersChanged = true;
            reset();
        }
        ImGui::SameLine();
        if (ImGui::Button("-") && state.numHiddenLayers > 0) {
            state.numHiddenLayers--;
            state.networkShape.pop_back();
            parametersChanged = true;
            reset();
        }

        for (int i = 0; i < state.numHiddenLayers; ++i) {
            std::string label = "Neurons in layer " + std::to_string(i + 1);
            ImGui::SliderInt(label.c_str(), &state.networkShape[i], 1, 8);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                parametersChanged = true;
                reset();
            }
        }
    }

    ImGui::Separator();

    // --- Output Section ---
    if (ImGui::CollapsingHeader("Output", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::string current_act_key = state.activationKey;
        if (ImGui::BeginCombo("Activation", current_act_key.c_str())) {
            for (auto const& [key, val] : activations) {
                if (ImGui::Selectable(key.c_str(), key == current_act_key)) {
                    state.activationKey = key;

                    parametersChanged = true;
                    reset();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::SliderFloat("Learning rate", &state.learningRate, 0.001f, 0.3f, "%.3f", ImGuiSliderFlags_Logarithmic);

        std::string current_reg_key = getKeyFromValue(regularizations, state.regularization);
        if (ImGui::BeginCombo("Regularization", current_reg_key.c_str())) {
            for (auto const& [key, val] : regularizations) {
                if (ImGui::Selectable(key.c_str(), key == current_reg_key)) {
                    state.regularization = val;
                    parametersChanged = true;
                    reset();
                }
            }
            ImGui::EndCombo();
        }
        ImGui::SliderFloat("Regularization rate", &state.regularizationRate, 0.0f, 0.3f, "%.2f");

        ImGui::Separator();
        ImGui::Text("Model Information:");
        // Input Features
        std::string input_features_str = "Input Features: ";
        auto input_ids = constructInputIds();
        for (size_t i = 0; i < input_ids.size(); ++i) {
            input_features_str += INPUTS[input_ids[i]].label;
            if (i < input_ids.size() - 1) {
                input_features_str += ", ";
            }
        }
        ImGui::TextWrapped("%s", input_features_str.c_str());

        // Network Shape
        std::string network_shape_str = "Network Shape: ";
        network_shape_str += std::to_string(input_ids.size()); // Input layer
        for (int i = 0; i < state.numHiddenLayers; ++i) {
            network_shape_str += " -> " + std::to_string(state.networkShape[i]);
        }
        network_shape_str += " -> 1"; // Output layer
        ImGui::Text("%s", network_shape_str.c_str());
    }
}

void PlaygroundApp::drawNetwork() {
    const float RECT_SIZE = 30.0f;
    const float PADDING = 20.0f;

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImVec2 size = ImGui::GetContentRegionAvail();

    // Calculate node positions
    node2coord.clear();
    int numLayers = network.size();
    float layer_x_step = (size.x - 2 * PADDING - RECT_SIZE) / (numLayers - 1);

    // Input layer
    auto input_ids = constructInputIds();
    float node_y_step_input = (size.y - 2 * PADDING) / (input_ids.size() + 1);
    for (size_t i = 0; i < input_ids.size(); ++i) {
        node2coord[input_ids[i]] = ImVec2(p.x + PADDING, p.y + PADDING + (i + 1) * node_y_step_input);
    }

    // Hidden and output layers
    for (int i = 1; i < numLayers; ++i) {
        float node_y_step = (size.y - 2 * PADDING) / (network[i].size() + 1);
        for (size_t j = 0; j < network[i].size(); ++j) {
            node2coord[network[i][j]->id] = ImVec2(p.x + PADDING + i * layer_x_step, p.y + PADDING + (j + 1) * node_y_step);
        }
    }

    // Draw links
    for (const auto& layer : network) {
        for (const auto& node : layer) {
            for (const auto& link : node->inputLinks) {
                ImVec2 p1 = node2coord[link->source->id];
                ImVec2 p2 = node2coord[link->dest->id];
                float weight_abs = std::abs(link->weight);
                ImU32 color = mainHeatMap.getColor(link->weight / 2.0); // Scale weight for color
                drawList->AddLine(p1, p2, color, 1.0f + weight_abs * 1.5f);
            }
        }
    }

    // Draw nodes
    for (const auto& [id, pos] : node2coord) {
        drawList->AddRectFilled(ImVec2(pos.x - RECT_SIZE/2, pos.y - RECT_SIZE/2), ImVec2(pos.x + RECT_SIZE/2, pos.y + RECT_SIZE/2), IM_COL32(255,255,255,255), 4.0f);
        drawList->AddRect(ImVec2(pos.x - RECT_SIZE/2, pos.y - RECT_SIZE/2), ImVec2(pos.x + RECT_SIZE/2, pos.y + RECT_SIZE/2), IM_COL32(0,0,0,255), 4.0f);
    }
}

#include <fstream>

void PlaygroundApp::drawOutput() {
    ImGui::Text("Test loss: %.3f", lossTest);
    ImGui::SameLine();
    ImGui::Text("Train loss: %.3f", lossTrain);
    if (state.showOverfit) {
        ImGui::SameLine();
        ImGui::Text("Overfit: %.3f", lossTrain - lossTest);
    }

    lineChart.draw();

    ImGui::Checkbox("Show test data", &state.showTestData);
    ImGui::SameLine();
    ImGui::Checkbox("Discretize output", &state.discretize);
    ImGui::SameLine();
    ImGui::Checkbox("Show data points", &state.showDataPoints);
    ImGui::SameLine();
    ImGui::Checkbox("Show potential overfit", &state.showOverfit);

    // Debugging: Display data sizes
    ImGui::Text("Train Data Size: %zu", trainData.size());
    ImGui::SameLine();
    ImGui::Text("Test Data Size: %zu", testData.size());

    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
    ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
    canvas_sz.x = std::min(canvas_sz.x, canvas_sz.y);
    canvas_sz.y = canvas_sz.x;
    ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRect(canvas_p0, canvas_p1, IM_COL32(512, 512, 512, 255));

    mainHeatMap.draw(drawList, canvas_p0, canvas_sz);

    if (state.showDataPoints) {
        mainHeatMap.drawDataPoints(drawList, canvas_p0, canvas_sz, trainData);
        if (state.showTestData) {
            mainHeatMap.drawDataPoints(drawList, canvas_p0, canvas_sz, testData);
        }
    }
}

void PlaygroundApp::reset(bool onStartup) {
    if (!onStartup) {
        // Change seed
    }

    if (!network.empty()) {
        nn::deleteNetwork(network);
    }

    lineChart.reset();
    iter = 0;

    auto inputIds = constructInputIds();
    std::vector<int> shape = { (int)inputIds.size() };
    shape.insert(shape.end(), state.networkShape.begin(), state.networkShape.end());
    shape.push_back(1);

    nn::ActivationFunction outputActivation = (state.problem == Problem::REGRESSION) ?
        nn::Activations::LINEAR : nn::Activations::TANH;

    network = nn::buildNetwork(shape, activations[state.activationKey], outputActivation, state.regularization, inputIds, state.initZero);

    // ================== FIX START ==================
    // Clear the old boundary data and initialize it for the new network.
    boundary.clear();
    nn::forEachNode(network, true, [this](nn::Node* node) {
        // For each node, create a 2D vector of the correct size.
        boundary[node->id] = std::vector<std::vector<double>>(DENSITY, std::vector<double>(DENSITY));
    });
    // =================== FIX END ===================

    generateData(onStartup);
    updateUIState();
}

void PlaygroundApp::oneStep() {
    iter++;
    for (size_t i = 0; i < trainData.size(); ++i) {
        auto& point = trainData[i];
        auto input = constructInput(point.x, point.y);
        nn::forwardProp(network, input);
        nn::backProp(network, point.label, nn::Errors::SQUARE);
        if ((i + 1) % state.batchSize == 0) {
            nn::updateWeights(network, state.learningRate, state.regularizationRate);
        }
    }
    updateUIState();
}

void PlaygroundApp::updateUIState() {
    lossTrain = getLoss(network, trainData);
    lossTest = getLoss(network, testData);
    lineChart.addDataPoint(lossTrain, lossTest);

    updateDecisionBoundary();
    mainHeatMap.updateBackground(boundary[nn::getOutputNode(network)->id], state.discretize);
}

void PlaygroundApp::generateData(bool firstTime) {
    int numSamples = state.numSamples;
    auto generator = (state.problem == Problem::CLASSIFICATION) ? state.dataset : state.regDataset;
    auto data = generator(numSamples, state.noise);

    playground::shuffle(data);

    int splitIndex = static_cast<int>(data.size() * state.percTrainData / 100.0);
    trainData = std::vector<playground::Example2D>(data.begin(), data.begin() + splitIndex);
    testData = std::vector<playground::Example2D>(data.begin() + splitIndex, data.end());
}

std::vector<std::string> PlaygroundApp::constructInputIds() {
    std::vector<std::string> result;
    if (state.x) result.push_back("x");
    if (state.y) result.push_back("y");
    if (state.xSquared) result.push_back("xSquared");
    if (state.ySquared) result.push_back("ySquared");
    if (state.xTimesY) result.push_back("xTimesY");
    if (state.sinX) result.push_back("sinX");
    return result;
}

std::vector<double> PlaygroundApp::constructInput(double x, double y) {
    std::vector<double> input;
    if (state.x) input.push_back(INPUTS["x"].f(x, y));
    if (state.y) input.push_back(INPUTS["y"].f(x, y));
    if (state.xSquared) input.push_back(INPUTS["xSquared"].f(x, y));
    if (state.ySquared) input.push_back(INPUTS["ySquared"].f(x, y));
    if (state.xTimesY) input.push_back(INPUTS["xTimesY"].f(x, y));
    if (state.sinX) input.push_back(INPUTS["sinX"].f(x, y));
    return input;
}

void PlaygroundApp::updateDecisionBoundary() {

    for (int i = 0; i < DENSITY; ++i) {
        for (int j = 0; j < DENSITY; ++j) {
            double x = map_range(i, 0, DENSITY - 1, xDomain.first, xDomain.second);
            double y = map_range(j, 0, DENSITY - 1, yDomain.first, yDomain.second);
            auto input = constructInput(x, y);
            nn::forwardProp(network, input);
            nn::forEachNode(network, true, [this, i, j](nn::Node* node) {

                boundary[node->id][i][j] = node->output;
            });
        }
    }
}

double PlaygroundApp::getLoss(nn::Network& net, const std::vector<playground::Example2D>& data) {
    if (data.empty()) return 0.0;
    double totalLoss = 0;
    for (const auto& point : data) {
        auto input = constructInput(point.x, point.y);
        double output = nn::forwardProp(net, input);
        totalLoss += nn::Errors::SQUARE.error(output, point.label);
    }
    return totalLoss / data.size();
}
