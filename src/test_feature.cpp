#include "playground.hpp"
#include <iostream>
#include <string>
#include <cmath>
#include <cassert>

// Helper for comparing floating point numbers
void assert_close(double a, double b, double epsilon = 1e-9, const std::string& msg = "") {
    if (std::abs(a - b) > epsilon) {
        std::cerr << "ASSERT FAILED: " << a << " is not close to " << b << ". " << msg << std::endl;
        assert(false);
    }
}

// --- Feature Definitions ---
struct InputFeature {
    std::function<double(double, double)> f;
    std::string label;
};
extern std::map<std::string, InputFeature> INPUTS;

void test_x_feature() {
    std::cout << "--- Running Test: X Feature ---" << std::endl;
    InputFeature feature = INPUTS["x"];
    assert_close(feature.f(1.0, 2.0), 1.0, 1e-9, "X feature failed for (1, 2)");
    assert_close(feature.f(-5.0, 10.0), -5.0, 1e-9, "X feature failed for (-5, 10)");
    assert_close(feature.f(0.0, 0.0), 0.0, 1e-9, "X feature failed for (0, 0)");
    std::cout << "PASSED" << std::endl << std::endl;
}

void test_y_feature() {
    std::cout << "--- Running Test: Y Feature ---" << std::endl;
    InputFeature feature = INPUTS["y"];
    assert_close(feature.f(1.0, 2.0), 2.0, 1e-9, "Y feature failed for (1, 2)");
    assert_close(feature.f(-5.0, 10.0), 10.0, 1e-9, "Y feature failed for (-5, 10)");
    assert_close(feature.f(0.0, 0.0), 0.0, 1e-9, "Y feature failed for (0, 0)");
    std::cout << "PASSED" << std::endl << std::endl;
}

void test_x_squared_feature() {
    std::cout << "--- Running Test: X Squared Feature ---" << std::endl;
    InputFeature feature = INPUTS["xSquared"];
    assert_close(feature.f(2.0, 3.0), 4.0, 1e-9, "X Squared feature failed for (2, 3)");
    assert_close(feature.f(-3.0, 5.0), 9.0, 1e-9, "X Squared feature failed for (-3, 5)");
    assert_close(feature.f(0.0, 0.0), 0.0, 1e-9, "X Squared feature failed for (0, 0)");
    std::cout << "PASSED" << std::endl << std::endl;
}

void test_y_squared_feature() {
    std::cout << "--- Running Test: Y Squared Feature ---" << std::endl;
    InputFeature feature = INPUTS["ySquared"];
    assert_close(feature.f(2.0, 3.0), 9.0, 1e-9, "Y Squared feature failed for (2, 3)");
    assert_close(feature.f(3.0, -4.0), 16.0, 1e-9, "Y Squared feature failed for (3, -4)");
    assert_close(feature.f(0.0, 0.0), 0.0, 1e-9, "Y Squared feature failed for (0, 0)");
    std::cout << "PASSED" << std::endl << std::endl;
}

void test_x_times_y_feature() {
    std::cout << "--- Running Test: X Times Y Feature ---" << std::endl;
    InputFeature feature = INPUTS["xTimesY"];
    assert_close(feature.f(2.0, 3.0), 6.0, 1e-9, "X Times Y feature failed for (2, 3)");
    assert_close(feature.f(-2.0, 5.0), -10.0, 1e-9, "X Times Y feature failed for (-2, 5)");
    assert_close(feature.f(0.0, 5.0), 0.0, 1e-9, "X Times Y feature failed for (0, 5)");
    std::cout << "PASSED" << std::endl << std::endl;
}

void test_sin_x_feature() {
    std::cout << "--- Running Test: Sin(X) Feature ---" << std::endl;
    InputFeature feature = INPUTS["sinX"];
    assert_close(feature.f(0.0, 5.0), 0.0, 1e-9, "Sin(X) feature failed for (0, 5)");
    assert_close(feature.f(M_PI / 2.0, 1.0), 1.0, 1e-9, "Sin(X) feature failed for (PI/2, 1)");
    assert_close(feature.f(M_PI, 2.0), 0.0, 1e-9, "Sin(X) feature failed for (PI, 2)");
    std::cout << "PASSED" << std::endl << std::endl;
}

int main() {
    try {
        test_x_feature();
        test_y_feature();
        test_x_squared_feature();
        test_y_squared_feature();
        test_x_times_y_feature();
        test_sin_x_feature();

        std::cout << "All feature tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "A test failed with an exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
