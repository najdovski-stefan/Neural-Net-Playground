#include "nn.hpp"
#include <random>
#include <cmath>
#include <stdexcept>
#include <set>
#include <chrono>

namespace nn {

// ==============================================================================
// UTILITY/HELPER FUNCTIONS
// ==============================================================================

// A helper to get a thread-safe random number generator, seeded once.
static std::mt19937& getRandomEngine() {
    static std::seed_seq ss{
        static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
        std::random_device{}(),
    };
    static std::mt19937 engine(ss);
    return engine;
}

// Returns a random number in [-0.5, 0.5).
static double randHalf() {
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    return dist(getRandomEngine());
}

// ==============================================================================
// FUNCTION DEFINITIONS
// ==============================================================================

const ActivationFunction Activations::TANH = {
    [](double x) { return std::tanh(x); },
    [](double x) {
        double output = std::tanh(x);
        return 1 - output * output;
    }
};

const ActivationFunction Activations::RELU = {
    [](double x) { return std::max(0.0, x); },
    [](double x) { return x <= 0 ? 0.0 : 1.0; }
};

const ActivationFunction Activations::SIGMOID = {
    [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
    [](double x) {
        double output = 1.0 / (1.0 + std::exp(-x));
        return output * (1.0 - output);
    }
};

const ActivationFunction Activations::LINEAR = {
    [](double x) { return x; },
    [](double x) { return 1.0; }
};

const RegularizationFunction RegularizationFunctions::L1 = {
    [](double w) { return std::abs(w); },
    [](double w) { return w < 0 ? -1.0 : (w > 0 ? 1.0 : 0.0); }
};

const RegularizationFunction RegularizationFunctions::L2 = {
    [](double w) { return 0.5 * w * w; },
    [](double w) { return w; }
};

const ErrorFunction Errors::SQUARE = {
    [](double output, double target) { return 0.5 * std::pow(output - target, 2); },
    [](double output, double target) { return output - target; }
};

// ==============================================================================
// CLASS IMPLEMENTATIONS
// ==============================================================================

Node::Node(std::string id, const ActivationFunction& activation, bool initZero)
    : id(std::move(id)), activation(activation) {
    if (initZero) {
        this->bias = 0;
    }
}

double Node::updateOutput() {
    totalInput = bias;
    for (const auto& link : inputLinks) {
        totalInput += link->weight * link->source->output;
    }
    output = activation.output(totalInput);
    return output;
}

Link::Link(Node* source, Node* dest, const RegularizationFunction* regularization, bool initZero)
    : source(source), dest(dest), regularization(regularization) {
    this->id = source->id + "-" + dest->id;
    this->weight = initZero ? 0 : randHalf();
}

// ==============================================================================
// NETWORK FUNCTIONS
// ==============================================================================

Network buildNetwork(
    const std::vector<int>& networkShape,
    const ActivationFunction& activation,
    const ActivationFunction& outputActivation,
    const RegularizationFunction* regularization,
    const std::vector<std::string>& inputIds,
    bool initZero) {

    int numLayers = networkShape.size();
    int idCounter = 1;
    Network network;

    for (int layerIdx = 0; layerIdx < numLayers; ++layerIdx) {
        bool isOutputLayer = layerIdx == numLayers - 1;
        bool isInputLayer = layerIdx == 0;

        std::vector<Node*> currentLayer;
        network.push_back(currentLayer);

        int numNodes = networkShape[layerIdx];
        for (int i = 0; i < numNodes; ++i) {
            std::string nodeId = isInputLayer ? inputIds[i] : std::to_string(idCounter++);

            Node* node = new Node(nodeId, isOutputLayer ? outputActivation : activation, initZero);
            network[layerIdx].push_back(node);

            if (layerIdx >= 1) {
                // Add links from nodes in the previous layer to this node.
                for (Node* prevNode : network[layerIdx - 1]) {
                    Link* link = new Link(prevNode, node, regularization, initZero);
                    prevNode->outputs.push_back(link);
                    node->inputLinks.push_back(link);
                }
            }
        }
    }
    return network;
}

void deleteNetwork(Network& network) {
    std::set<Link*> allLinks;
    for (const auto& layer : network) {
        for (Node* node : layer) {
            for (Link* link : node->inputLinks) {
                allLinks.insert(link);
            }
        }
    }
    for (Link* link : allLinks) {
        delete link;
    }
    for (const auto& layer : network) {
        for (Node* node : layer) {
            delete node;
        }
    }
    network.clear();
}

double forwardProp(Network& network, const std::vector<double>& inputs) {
    auto& inputLayer = network[0];
    if (inputs.size() != inputLayer.size()) {
        throw std::runtime_error("The number of inputs must match the number of nodes in the input layer");
    }
    // Update the input layer.
    for (size_t i = 0; i < inputLayer.size(); ++i) {
        inputLayer[i]->output = inputs[i];
    }
    // Update the rest of the layers.
    for (size_t layerIdx = 1; layerIdx < network.size(); ++layerIdx) {
        for (Node* node : network[layerIdx]) {
            node->updateOutput();
        }
    }
    return network.back()[0]->output;
}

void backProp(Network& network, double target, const ErrorFunction& errorFunc) {
    Node* outputNode = network.back()[0];
    outputNode->outputDer = errorFunc.der(outputNode->output, target);

    // Go through the layers backwards.
    for (int layerIdx = network.size() - 1; layerIdx >= 1; --layerIdx) {
        auto& currentLayer = network[layerIdx];

        // Compute derivatives for nodes in this layer.
        for (Node* node : currentLayer) {
            node->inputDer = node->outputDer * node->activation.der(node->totalInput);
            node->accInputDer += node->inputDer;
            node->numAccumulatedDers++;
        }

        // Compute derivatives for links coming into this layer.
        for (Node* node : currentLayer) {
            for (Link* link : node->inputLinks) {
                if (link->isDead) continue;
                link->errorDer = node->inputDer * link->source->output;
                link->accErrorDer += link->errorDer;
                link->numAccumulatedDers++;
            }
        }

        if (layerIdx == 1) continue;

        // Compute output derivatives for the previous layer.
        auto& prevLayer = network[layerIdx - 1];
        for (Node* node : prevLayer) {
            node->outputDer = 0;
            for (Link* outputLink : node->outputs) {
                node->outputDer += outputLink->weight * outputLink->dest->inputDer;
            }
        }
    }
}

void updateWeights(Network& network, double learningRate, double regularizationRate) {
    for (size_t layerIdx = 1; layerIdx < network.size(); ++layerIdx) {
        for (Node* node : network[layerIdx]) {
            // Update the node's bias.
            if (node->numAccumulatedDers > 0) {
                node->bias -= learningRate * node->accInputDer / node->numAccumulatedDers;
                node->accInputDer = 0;
                node->numAccumulatedDers = 0;
            }
            // Update the weights coming into this node.
            for (Link* link : node->inputLinks) {
                if (link->isDead) continue;

                if (link->numAccumulatedDers > 0) {
                    // Update the weight based on dE/dw.
                    link->weight -= (learningRate / link->numAccumulatedDers) * link->accErrorDer;

                    // Further update the weight based on regularization.
                    double regulDer = link->regularization ? link->regularization->der(link->weight) : 0;
                    double newLinkWeight = link->weight - (learningRate * regularizationRate) * regulDer;

                    if (link->regularization == &RegularizationFunctions::L1 && link->weight * newLinkWeight < 0) {
                        // The weight crossed 0 due to L1 regularization. Set it to 0.
                        link->weight = 0;
                        link->isDead = true;
                    } else {
                        link->weight = newLinkWeight;
                    }

                    link->accErrorDer = 0;
                    link->numAccumulatedDers = 0;
                }
            }
        }
    }
}

void forEachNode(Network& network, bool ignoreInputs, std::function<void(Node*)> accessor) {
    for (size_t layerIdx = ignoreInputs ? 1 : 0; layerIdx < network.size(); ++layerIdx) {
        for (Node* node : network[layerIdx]) {
            accessor(node);
        }
    }
}

Node* getOutputNode(Network& network) {
    return network.back()[0];
}

std::map<std::string, const RegularizationFunction*> regularizations = {
    {"none", nullptr},
    {"L1", &RegularizationFunctions::L1},
    {"L2", &RegularizationFunctions::L2}
};

} // namespace nn
