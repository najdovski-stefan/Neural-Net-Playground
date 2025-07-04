#include "linechart.hpp"
#include <imgui.h> // Include for ImVec4

LineChart::LineChart() {
    reset();
}

void LineChart::addDataPoint(double trainLoss, double testLoss) {
    trainLossData.push_back(static_cast<float>(trainLoss));
    testLossData.push_back(static_cast<float>(testLoss));
    x_max += 1.0f;
}

void LineChart::reset() {
    trainLossData.clear();
    testLossData.clear();
    x_max = 0;
}

void LineChart::draw() {
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0,0));
    if (ImPlot::BeginPlot("##Loss", ImVec2(-1, 55), ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoTitle)) {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, x_max > 0 ? x_max : 1, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.0);

        if (!testLossData.empty()) {
            // FIX: Use ImVec4 for the color instead of IM_COL32
            ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.0f, 0.0f, 1.0f)); // Black
            ImPlot::PlotLine("Test loss", testLossData.data(), testLossData.size());
        }
        if (!trainLossData.empty()) {
            // FIX: Use ImVec4 for the color instead of IM_COL32
            // 119/255 is approximately 0.467
            ImPlot::SetNextLineStyle(ImVec4(0.467f, 0.467f, 0.467f, 1.0f)); // Gray
            ImPlot::PlotLine("Train loss", trainLossData.data(), trainLossData.size());
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
}
