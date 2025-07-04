#include "dataset.hpp"
#include <iostream>
#include <string>
#include <algorithm> // For std::min

// Helper to print a few examples from a dataset
void print_examples(const std::string& name, const std::vector<playground::Example2D>& data) {
    std::cout << "--- " << name << " ---" << std::endl;
    std::cout << "Generated " << data.size() << " points." << std::endl;
    for (size_t i = 0; i < std::min(data.size(), static_cast<size_t>(5)); ++i) {
        const auto& p = data[i];
        std::cout << "  Point " << i << ": (x=" << p.x << ", y=" << p.y << ", label=" << p.label << ")" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // Test classifyTwoGaussData
    {
        int num_samples_test = 10;
        double noise_test = 0.1;
        auto test_data = playground::classifyTwoGaussData(num_samples_test, noise_test);
        if (test_data.size() != num_samples_test) {
            std::cerr << "Test Failed: classifyTwoGaussData generated incorrect number of samples." << std::endl;
        } else {
            std::cout << "Test Passed: classifyTwoGaussData" << std::endl;
        }
    }

    // Test classifySpiralData
    {
        int num_samples_test = 15;
        double noise_test = 0.15;
        auto test_data = playground::classifySpiralData(num_samples_test, noise_test);
        if (test_data.size() != num_samples_test) {
            std::cerr << "Test Failed: classifySpiralData generated incorrect number of samples." << std::endl;
        } else {
            std::cout << "Test Passed: classifySpiralData" << std::endl;
        }
    }

    // Test shuffle
    {
        std::vector<playground::Example2D> test_data = {{ {1.0, 1.0, 0}, {2.0, 2.0, 1}, {3.0, 3.0, 0} }};
        std::vector<playground::Example2D> original_data = test_data;
        playground::shuffle(test_data);
        bool shuffled = false;
        if (test_data.size() == original_data.size()) {
            for (size_t i = 0; i < test_data.size(); ++i) {
                if (test_data[i].x != original_data[i].x || test_data[i].y != original_data[i].y || test_data[i].label != original_data[i].label) {
                    shuffled = true;
                    break;
                }
            }
        }
        if (shuffled) {
            std::cout << "Test Passed: shuffle" << std::endl;
        } else {
            std::cerr << "Test Failed: shuffle did not reorder the data." << std::endl;
        }
    }

    // Test classifyXORData
    {
        int num_samples_test = 20;
        double noise_test = 0.05;
        auto test_data = playground::classifyXORData(num_samples_test, noise_test);
        if (test_data.size() != num_samples_test) {
            std::cerr << "Test Failed: classifyXORData generated incorrect number of samples." << std::endl;
        } else {
            std::cout << "Test Passed: classifyXORData" << std::endl;
        }
    }

    // Test classifyCircleData
    {
        int num_samples_test = 25;
        double noise_test = 0.25;
        auto test_data = playground::classifyCircleData(num_samples_test, noise_test);
        if (test_data.size() != num_samples_test) {
            std::cerr << "Test Failed: classifyCircleData generated incorrect number of samples." << std::endl;
        } else {
            std::cout << "Test Passed: classifyCircleData" << std::endl;
        }
    }


    return 0;
}
