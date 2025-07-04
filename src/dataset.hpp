#pragma once

#include <vector>
#include <functional>

namespace playground {

/**
 * A two dimensional example: x and y coordinates with the label.
 */
struct Example2D {
    double x;
    double y;
    double label;
};

/**
 * A simple 2D point.
 */
struct Point {
    double x;
    double y;
};

/**
 * Shuffles the vector using the Fisher-Yates algorithm provided by the standard library.
 */
void shuffle(std::vector<Example2D>& array);

/**
 * A function that generates data.
 */
using DataGenerator = std::function<std::vector<Example2D>(int numSamples, double noise)>;

// Data generation functions
std::vector<Example2D> classifyTwoGaussData(int numSamples, double noise);
std::vector<Example2D> regressPlane(int numSamples, double noise);
std::vector<Example2D> regressGaussian(int numSamples, double noise);
std::vector<Example2D> classifySpiralData(int numSamples, double noise);
std::vector<Example2D> classifyCircleData(int numSamples, double noise);
std::vector<Example2D> classifyXORData(int numSamples, double noise);
std::vector<Example2D> classifyStarData(int numSamples, double noise);
std::vector<Example2D> classifySineData(int numSamples, double noise);
std::vector<Example2D> classifyCheckerboardData(int numSamples, double noise);
std::vector<Example2D> classifyMoonsData(int numSamples, double noise);
std::vector<Example2D> classifyHeartData(int numSamples, double noise);

} // namespace playground
