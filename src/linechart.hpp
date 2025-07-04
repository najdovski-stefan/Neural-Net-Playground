#pragma once

#include <vector>
#include <implot.h>

class LineChart {
public:
    LineChart();
    void addDataPoint(double trainLoss, double testLoss);
    void reset();
    void draw();

private:
    std::vector<float> trainLossData;
    std::vector<float> testLossData;
    float x_max = 0;
};
