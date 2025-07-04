#pragma once

#include <vector>
#include <string>
#include <functional>
#include <map>

namespace nn {

// Forward declarations for graph structure
struct Node;
struct Link;

// Functional interfaces
struct ActivationFunction {
    std::function<double(double)> output;
    std::function<double(double)> der;
};

struct RegularizationFunction {
    std::function<double(double)> output;
    std::function<double(double)> der;
};

struct ErrorFunction {
    std::function<double(double, double)> error;
    std::function<double(double, double)> der;
};

// Static instances of available functions
struct Activations {
    static const ActivationFunction TANH;
    static const ActivationFunction RELU;
    static const ActivationFunction SIGMOID;
    static const ActivationFunction LINEAR;
};

struct RegularizationFunctions {
    static const RegularizationFunction L1;
    static const RegularizationFunction L2;
};

struct Errors {
    static const ErrorFunction SQUARE;
};

/**
 * A node in a neural network.
 */
struct Node {
    std::string id;
    std::vector<Link*> inputLinks;
    std::vector<Link*> outputs;
    double bias = 0.1;
    double totalInput = 0.0;
    double output = 0.0;
    double outputDer = 0.0;
    double inputDer = 0.0;
    double accInputDer = 0.0;
    int numAccumulatedDers = 0;
    ActivationFunction activation;

    Node(std::string id, const ActivationFunction& activation, bool initZero = false);
    double updateOutput();
};

/**
 * A link in a neural network.
 */
struct Link {
    std::string id;
    Node* source;
    Node* dest;
    double weight;
    bool isDead = false;
    double errorDer = 0.0;
    double accErrorDer = 0.0;
    int numAccumulatedDers = 0;
    const RegularizationFunction* regularization;

    Link(Node* source, Node* dest, const RegularizationFunction* regularization, bool initZero = false);
};

// Type alias for the network structure
using Network = std::vector<std::vector<Node*>>;

// --- Core Network Functions ---

/**
 * Builds a neural network.
 * IMPORTANT: The returned network must be freed using `deleteNetwork` to avoid memory leaks.
 */
Network buildNetwork(
    const std::vector<int>& networkShape,
    const ActivationFunction& activation,
    const ActivationFunction& outputActivation,
    const RegularizationFunction* regularization,
    const std::vector<std::string>& inputIds,
    bool initZero = false
);

/**
 * Frees the memory allocated by `buildNetwork`.
 */
void deleteNetwork(Network& network);

/**
 * Runs a forward propagation of the provided input through the network.
 */
double forwardProp(Network& network, const std::vector<double>& inputs);

/**
 * Runs a backward propagation using the provided target.
 */
void backProp(Network& network, double target, const ErrorFunction& errorFunc);

/**
 * Updates the weights of the network using accumulated error derivatives.
 */
void updateWeights(Network& network, double learningRate, double regularizationRate);


// --- Utility Functions ---

/**
 * Iterates over every node in the network.
 */
void forEachNode(Network& network, bool ignoreInputs, std::function<void(Node*)> accessor);

/**
 * Returns the output node in the network.
 */
Node* getOutputNode(Network& network);

/**
 * A map to retrieve regularization functions by name.
 * "none" maps to nullptr.
 */
extern std::map<std::string, const RegularizationFunction*> regularizations;

} // namespace nn
