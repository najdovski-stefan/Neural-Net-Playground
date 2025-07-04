#include "dataset.hpp"

#include <random>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <chrono> // Required for a better random seed

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace playground {

// ==============================================================================
// UTILITY/HELPER FUNCTIONS
// ==============================================================================

// A helper to get a thread-safe random number generator, seeded once.
// This version uses a more robust seeding mechanism.
static std::mt19937& getRandomEngine() {
    // Seed with a combination of a random device and the current time to ensure
    // high-quality randomness and prevent deterministic behavior on some platforms.
    static std::seed_seq ss{
        static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
        std::random_device{}(),
        std::random_device{}(),
        std::random_device{}(),
    };
    static std::mt19937 engine(ss);
    return engine;
}

/**
 * Returns a sample from a uniform [a, b] distribution.
 */
static double randUniform(double a, double b) {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(getRandomEngine());
}

/**
 * Samples from a normal distribution.
 * @param mean The mean.
 * @param variance The variance.
 */
static double normalRandom(double mean = 0.0, double variance = 1.0) {
    std::normal_distribution<double> dist(mean, std::sqrt(variance));
    return dist(getRandomEngine());
}

/**
 * Returns the Euclidean distance between two points.
 */
static double dist(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * A C++ implementation of d3.scale.linear.
 * Maps a value from a domain [min, max] to a range [min, max].
 */
static double linearScale(double value, double domainMin, double domainMax, double rangeMin, double rangeMax, bool clamp = false) {
    if (clamp) {
        value = std::max(domainMin, std::min(domainMax, value));
    }
    if (domainMax == domainMin) {
        return rangeMin;
    }
    double ratio = (value - domainMin) / (domainMax - domainMin);
    return rangeMin + ratio * (rangeMax - rangeMin);
}


// ==============================================================================
// PUBLIC API IMPLEMENTATION
// ==============================================================================

void shuffle(std::vector<Example2D>& array) {
    std::shuffle(array.begin(), array.end(), getRandomEngine());
}

std::vector<Example2D> classifyTwoGaussData(int numSamples, double noise) {
    std::vector<Example2D> points;
    points.reserve(numSamples);

    double variance = linearScale(noise, 0.0, 0.5, 0.5, 4.0);

    // Correctly split samples to handle odd numSamples.
    const int n1 = numSamples / 2;
    const int n2 = numSamples - n1;

    auto genGauss = [&](int count, double cx, double cy, double label) {
        for (int i = 0; i < count; i++) {
            double x = normalRandom(cx, variance);
            double y = normalRandom(cy, variance);
            points.push_back({x, y, label});
        }
    };

    genGauss(n1, 2, 2, 1);  // Gaussian with positive examples.
    genGauss(n2, -2, -2, -1); // Gaussian with negative examples.
    return points;
}

std::vector<Example2D> regressPlane(int numSamples, double noise) {
    double radius = 6.0;

    auto getLabel = [&](double x, double y) {
        return linearScale(x + y, -10.0, 10.0, -1.0, 1.0);
    };

    std::vector<Example2D> points;
    points.reserve(numSamples);
    for (int i = 0; i < numSamples; i++) {
        double x = randUniform(-radius, radius);
        double y = randUniform(-radius, radius);
        double noiseX = randUniform(-radius, radius) * noise;
        double noiseY = randUniform(-radius, radius) * noise;
        double label = getLabel(x + noiseX, y + noiseY);
        points.push_back({x, y, label});
    }
    return points;
}

std::vector<Example2D> regressGaussian(int numSamples, double noise) {
    std::vector<Example2D> points;
    points.reserve(numSamples);

    static const std::vector<std::array<double, 3>> gaussians = {
        {-4, 2.5, 1}, {0, 2.5, -1}, {4, 2.5, 1},
        {-4, -2.5, -1}, {0, -2.5, 1}, {4, -2.5, -1}
    };

    auto getLabel = [&](double x, double y) {
        double maxAbsLabel = 0.0;
        double finalLabel = 0.0;
        for (const auto& g : gaussians) {
            double cx = g[0], cy = g[1], sign = g[2];
            double d = dist({x, y}, {cx, cy});
            double newLabel = sign * linearScale(d, 0.0, 2.0, 1.0, 0.0, true);
            if (std::abs(newLabel) > maxAbsLabel) {
                maxAbsLabel = std::abs(newLabel);
                finalLabel = newLabel;
            }
        }
        return finalLabel;
    };

    double radius = 6.0;
    for (int i = 0; i < numSamples; i++) {
        double x = randUniform(-radius, radius);
        double y = randUniform(-radius, radius);
        double noiseX = randUniform(-radius, radius) * noise;
        double noiseY = randUniform(-radius, radius) * noise;
        double label = getLabel(x + noiseX, y + noiseY);
        points.push_back({x, y, label});
    }
    return points;
}

std::vector<Example2D> classifySpiralData(int numSamples, double noise) {
    std::vector<Example2D> points;
    points.reserve(numSamples);

    // Correctly split samples to handle odd numSamples.
    const int n_pos = numSamples / 2;
    const int n_neg = numSamples - n_pos;

    auto genSpiral = [&](int num_points_in_spiral, double deltaT, double label) {
        for (int i = 0; i < num_points_in_spiral; i++) {
            double r = static_cast<double>(i) / num_points_in_spiral * 5.0;
            double t = 1.75 * static_cast<double>(i) / num_points_in_spiral * 2.0 * M_PI + deltaT;
            double x = r * std::sin(t) + randUniform(-1, 1) * noise;
            double y = r * std::cos(t) + randUniform(-1, 1) * noise;
            points.push_back({x, y, label});
        }
    };

    genSpiral(n_pos, 0, 1); // Positive examples.
    genSpiral(n_neg, M_PI, -1); // Negative examples.
    return points;
}

std::vector<Example2D> classifyCircleData(int numSamples, double noise) {
    std::vector<Example2D> points;
    points.reserve(numSamples);
    double radius = 5.0;

    auto getCircleLabel = [&](const Point& p) {
        return (dist(p, {0, 0}) < (radius * 0.5)) ? 1.0 : -1.0;
    };

    // Correctly split samples to handle odd numSamples.
    const int num_positive = numSamples / 2;
    const int num_negative = numSamples - num_positive;

    // Generate positive points inside the circle.
    for (int i = 0; i < num_positive; i++) {
        double r = randUniform(0, radius * 0.5);
        double angle = randUniform(0, 2 * M_PI);
        double x = r * std::sin(angle);
        double y = r * std::cos(angle);
        double noiseX = randUniform(-radius, radius) * noise;
        double noiseY = randUniform(-radius, radius) * noise;
        double label = getCircleLabel({x + noiseX, y + noiseY});
        points.push_back({x, y, label});
    }

    // Generate negative points outside the circle.
    for (int i = 0; i < num_negative; i++) {
        double r = randUniform(radius * 0.7, radius);
        double angle = randUniform(0, 2 * M_PI);
        double x = r * std::sin(angle);
        double y = r * std::cos(angle);
        double noiseX = randUniform(-radius, radius) * noise;
        double noiseY = randUniform(-radius, radius) * noise;
        double label = getCircleLabel({x + noiseX, y + noiseY});
        points.push_back({x, y, label});
    }
    return points;
}

std::vector<Example2D> classifyXORData(int numSamples, double noise) {
    auto getXORLabel = [](const Point& p) {
        return p.x * p.y >= 0 ? 1.0 : -1.0;
    };

    std::vector<Example2D> points;
    points.reserve(numSamples);
    for (int i = 0; i < numSamples; i++) {
        double x = randUniform(-5, 5);
        double padding = 0.3;
        x += x > 0 ? padding : -padding;
        double y = randUniform(-5, 5);
        y += y > 0 ? padding : -padding;
        double noiseX = randUniform(-5, 5) * noise;
        double noiseY = randUniform(-5, 5) * noise;
        double label = getXORLabel({x + noiseX, y + noiseY});
        points.push_back({x, y, label});
    }
    return points;
}

std::vector<Example2D> classifyStarData(int numSamples, double noise) {
    std::vector<Example2D> points;
    points.reserve(numSamples);
    double radius = 5.0;

    auto getStarLabel = [&](const Point& p) {
        double angle = atan2(p.y, p.x) + M_PI;
        double r = sqrt(p.x * p.x + p.y * p.y);

        int num_points = 5;
        double a = M_PI / num_points;
        double t = fmod(angle, 2 * a);
        double r_star = radius / 2.0 * (cos(a) / cos(t - a));

        return (r < r_star) ? 1.0 : -1.0;
    };

    for (int i = 0; i < numSamples; i++) {
        double r = randUniform(0, radius);
        double angle = randUniform(0, 2 * M_PI);
        double x = r * std::sin(angle);
        double y = r * std::cos(angle);
        double noiseX = randUniform(-radius, radius) * noise;
        double noiseY = randUniform(-radius, radius) * noise;
        double label = getStarLabel({x + noiseX, y + noiseY});
        points.push_back({x, y, label});
    }
    return points;
}

std::vector<Example2D> classifySineData(int numSamples, double noise) {
    std::vector<Example2D> points;
    points.reserve(numSamples);
    double radius = 5.0;

    auto getSineLabel = [&](const Point& p) {
        return p.y > sin(p.x * 2.0) ? 1.0 : -1.0;
    };

    for (int i = 0; i < numSamples; i++) {
        double x = randUniform(-radius, radius);
        double y = randUniform(-radius, radius);
        double noiseX = randUniform(-radius, radius) * noise;
        double noiseY = randUniform(-radius, radius) * noise;
        double label = getSineLabel({x + noiseX, y + noiseY});
        points.push_back({x, y, label});
    }
    return points;
}

std::vector<Example2D> classifyCheckerboardData(int numSamples, double noise) {
    std::vector<Example2D> points;
    points.reserve(numSamples);
    double radius = 5.0;

    auto getCheckerboardLabel = [&](const Point& p) {
        return (static_cast<int>(floor(p.x / 2.0)) + static_cast<int>(floor(p.y / 2.0))) % 2 == 0 ? 1.0 : -1.0;
    };

    for (int i = 0; i < numSamples; i++) {
        double x = randUniform(-radius, radius);
        double y = randUniform(-radius, radius);
        double noiseX = randUniform(-radius, radius) * noise;
        double noiseY = randUniform(-radius, radius) * noise;
        double label = getCheckerboardLabel({x + noiseX, y + noiseY});
        points.push_back({x, y, label});
    }
    return points;
}

// Idea: Two interleaving half-circles (classic for testing classifiers).
std::vector<Example2D> classifyMoonsData(int numSamples, double noise) {
    std::vector<Example2D> points;
    points.reserve(numSamples);
    double radius = 4.0; // Reduced radius to fit within the domain
    double crescent_width = 2.0;

    auto getMoonLabel = [&](const Point& p) {
        double dist_to_center = dist(p, {0, 0});
        double dist_to_offset = dist(p, {-crescent_width / 2, 0});
        return (dist_to_center < radius && dist_to_offset > radius) ? 1.0 : -1.0;
    };

    for (int i = 0; i < numSamples; i++) {
        double x = randUniform(-6, 6);
        double y = randUniform(-6, 6);
        double noiseX = randUniform(-1, 1) * noise * 5.0;
        double noiseY = randUniform(-1, 1) * noise * 5.0;
        double label = getMoonLabel({x + noiseX, y + noiseY});
        points.push_back({x, y, label});
    }

    return points;
}

std::vector<Example2D> classifyHeartData(int numSamples, double noise) {
    std::vector<Example2D> points;
    points.reserve(numSamples);
    double radius = 6.0;

    auto getHeartLabel = [&](Point p) {
        p.y *= -1; // Flip the y-axis
        double x = p.x / (radius / 2.0);
        double y = p.y / (radius / 2.0);
        double x2 = x * x;
        double y2 = y * y;
        // Equation for a heart shape
        return (x2 + y2 - 1) * (x2 + y2 - 1) * (x2 + y2 - 1) - x2 * y2 * y < 0 ? 1.0 : -1.0;
    };

    for (int i = 0; i < numSamples; i++) {
        double x = randUniform(-radius, radius);
        double y = randUniform(-radius, radius);
        double noiseX = randUniform(-radius, radius) * noise;
        double noiseY = randUniform(-radius, radius) * noise;
        double label = getHeartLabel({x + noiseX, y + noiseY});
        points.push_back({x, y, label});
    }
    return points;
}

} // namespace playground
