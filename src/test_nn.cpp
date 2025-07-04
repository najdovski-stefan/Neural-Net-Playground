#include "nn.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <numeric>

// Helper for comparing floating point numbers
void assert_close(double a, double b, double epsilon = 1e-9, const std::string& msg = "") {
    if (std::abs(a - b) > epsilon) {
        std::cerr << "ASSERT FAILED: " << a << " is not close to " << b << ". " << msg << std::endl;
        assert(false);
    }
}

/**
 * Tests if the network is built with the correct structure and can be deleted.
 */
void test_build_and_delete_network() {
    std::cout << "--- Running Test: Build and Delete Network ---" << std::endl;

    std::vector<int> shape = {2, 3, 1};
    std::vector<std::string> input_ids = {"x1", "x2"};

    nn::Network network = nn::buildNetwork(shape, nn::Activations::TANH, nn::Activations::TANH, &nn::RegularizationFunctions::L2, input_ids);

    // Test network shape
    assert(network.size() == 3);
    assert(network[0].size() == 2); // Input layer
    assert(network[1].size() == 3); // Hidden layer
    assert(network[2].size() == 1); // Output layer

    // Test node IDs
    assert(network[0][0]->id == "x1");
    assert(network[0][1]->id == "x2");

    // Test link creation
    // Each node in the hidden layer should have 2 input links (from the input layer)
    assert(network[1][0]->inputLinks.size() == 2);
    // The output node should have 3 input links (from the hidden layer)
    assert(network[2][0]->inputLinks.size() == 3);
    // Each node in the input layer should have 3 output links (to the hidden layer)
    assert(network[0][0]->outputs.size() == 3);

    // Test if deletion works without crashing
    nn::deleteNetwork(network);
    assert(network.empty());

    std::cout << "PASSED" << std::endl << std::endl;
}

/**
 * Tests the forward propagation logic with known weights.
 */
void test_forward_propagation() {
    std::cout << "--- Running Test: Forward Propagation ---" << std::endl;

    std::vector<int> shape = {2, 1};
    std::vector<std::string> input_ids = {"x1", "x2"};

    // Use initZero to have predictable weights (0), but bias will be default (0.1)
    nn::Network network = nn::buildNetwork(shape, nn::Activations::LINEAR, nn::Activations::LINEAR, nullptr, input_ids);

    // Manually set weights and bias for deterministic calculation
    nn::Node* output_node = network[1][0];
    output_node->bias = 0.5;
    output_node->inputLinks[0]->weight = 0.2; // Link from input 0
    output_node->inputLinks[1]->weight = 0.3; // Link from input 1

    std::vector<double> inputs = {1.0, 2.0};
    double output = nn::forwardProp(network, inputs);

    // Manual calculation:
    // totalInput = bias + (input1 * weight1) + (input2 * weight2)
    //            = 0.5  + (1.0 * 0.2)      + (2.0 * 0.3)
    //            = 0.5  + 0.2              + 0.6
    //            = 1.3
    // output = LINEAR(1.3) = 1.3
    assert_close(output, 1.3, 1e-9, "Forward prop calculation is incorrect.");

    nn::deleteNetwork(network);
    std::cout << "PASSED" << std::endl << std::endl;
}

/**
 * Tests backpropagation and weight updates with a simple network.
 */
void test_backprop_and_update() {
    std::cout << "--- Running Test: Backpropagation and Weight Update ---" << std::endl;

    std::vector<int> shape = {1, 1};
    std::vector<std::string> input_ids = {"x"};
    nn::Network network = nn::buildNetwork(shape, nn::Activations::LINEAR, nn::Activations::LINEAR, nullptr, input_ids);

    // Setup a deterministic network state
    nn::Node* output_node = network[1][0];
    nn::Link* link = output_node->inputLinks[0];
    output_node->bias = 0.5;
    link->weight = 0.8;

    // 1. Forward pass
    std::vector<double> inputs = {2.0};
    double output = nn::forwardProp(network, inputs);
    // Manual calculation: output = bias + input * weight = 0.5 + 2.0 * 0.8 = 0.5 + 1.6 = 2.1
    assert_close(output, 2.1);

    // 2. Backward pass
    double target = 2.5;
    nn::backProp(network, target, nn::Errors::SQUARE);

    // Manual gradient calculation:
    // error_der = output - target = 2.1 - 2.5 = -0.4
    // input_der = error_der * activation_der(total_input) = -0.4 * 1.0 = -0.4
    // link_error_der = input_der * source_output = -0.4 * 2.0 = -0.8
    assert_close(output_node->outputDer, -0.4, 1e-9, "Output derivative is wrong.");
    assert_close(output_node->inputDer, -0.4, 1e-9, "Input derivative (bias gradient) is wrong.");
    assert_close(link->errorDer, -0.8, 1e-9, "Link error derivative (weight gradient) is wrong.");

    // 3. Update weights
    double learning_rate = 0.1;
    nn::updateWeights(network, learning_rate, 0.0); // No regularization

    // Manual update calculation:
    // new_bias = old_bias - lr * input_der = 0.5 - 0.1 * (-0.4) = 0.5 + 0.04 = 0.54
    // new_weight = old_weight - lr * link_error_der = 0.8 - 0.1 * (-0.8) = 0.8 + 0.08 = 0.88
    assert_close(output_node->bias, 0.54, 1e-9, "Bias update is wrong.");
    assert_close(link->weight, 0.88, 1e-9, "Weight update is wrong.");

    nn::deleteNetwork(network);
    std::cout << "PASSED" << std::endl << std::endl;
}

/**
 * An end-to-end test to see if the network can learn the XOR problem.
 */
void test_full_training_loop_XOR() {
    std::cout << "--- Running Test: Full Training Loop (XOR) ---" << std::endl;

    // XOR data: {input1, input2}, {target}
    std::vector<std::pair<std::vector<double>, double>> xor_data = {
        {{0.0, 0.0}, 0.0},
        {{0.0, 1.0}, 1.0},
        {{1.0, 0.0}, 1.0},
        {{1.0, 1.0}, 0.0}
    };

    // Build a network capable of learning XOR
    nn::Network network = nn::buildNetwork({2, 3, 1}, nn::Activations::TANH, nn::Activations::TANH, nullptr, {"x1", "x2"});

    double learning_rate = 0.1;
    int epochs = 2000;

    // Training loop
    for (int i = 0; i < epochs; ++i) {
        double total_error = 0;
        for (const auto& data_point : xor_data) {
            double output = nn::forwardProp(network, data_point.first);
            total_error += nn::Errors::SQUARE.error(output, data_point.second);
            nn::backProp(network, data_point.second, nn::Errors::SQUARE);
        }
        nn::updateWeights(network, learning_rate, 0.0);

        if ((i + 1) % 500 == 0) {
            std::cout << "Epoch " << i + 1 << ", Avg Error: " << total_error / xor_data.size() << std::endl;
        }
    }

    // Test the trained network
    std::cout << "Testing trained network..." << std::endl;
    double out1 = nn::forwardProp(network, {0.0, 0.0});
    double out2 = nn::forwardProp(network, {0.0, 1.0});
    double out3 = nn::forwardProp(network, {1.0, 0.0});
    double out4 = nn::forwardProp(network, {1.0, 1.0});

    std::cout << "[0,0] -> " << out1 << " (target 0)" << std::endl;
    std::cout << "[0,1] -> " << out2 << " (target 1)" << std::endl;
    std::cout << "[1,0] -> " << out3 << " (target 1)" << std::endl;
    std::cout << "[1,1] -> " << out4 << " (target 0)" << std::endl;

    assert(out1 < 0.5);
    assert(out2 > 0.5);
    assert(out3 > 0.5);
    assert(out4 < 0.5);

    nn::deleteNetwork(network);
    std::cout << "PASSED" << std::endl << std::endl;
}


int main() {
    try {
        test_build_and_delete_network();
        test_forward_propagation();
        test_backprop_and_update();
        test_full_training_loop_XOR();

        std::cout << "All tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "A test failed with an exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
