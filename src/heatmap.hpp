#pragma once

#include "dataset.hpp"
#include <imgui.h>
#include <vector>

class HeatMap {
public:
    HeatMap(int resolution, const std::pair<double, double>& xDomain, const std::pair<double, double>& yDomain);

    void updateBackground(const std::vector<std::vector<double>>& data, bool discretize);
    void draw(ImDrawList* drawList, ImVec2 canvas_p0, ImVec2 canvas_sz);
    void drawDataPoints(ImDrawList* drawList, ImVec2 canvas_p0, ImVec2 canvas_sz, const std::vector<playground::Example2D>& dataPoints);

public:
    ImU32 getColor(double value, bool opaque = false);
    ImVec2 scale(double x, double y, ImVec2 p0, ImVec2 p1);

private:
    int resolution;

public:
    std::pair<double, double> xDomain, yDomain;

private:
    std::vector<std::vector<ImU32>> backgroundColors;
};
