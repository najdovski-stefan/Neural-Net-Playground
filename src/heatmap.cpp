#include "heatmap.hpp"
#include <algorithm>
#include <iostream> // For debug logging

double map_range(double val, double in_min, double in_max, double out_min, double out_max) {
    if (in_max == in_min) {
        return out_min;
    }
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}


HeatMap::HeatMap(int resolution, const std::pair<double, double>& xDomain, const std::pair<double, double>& yDomain)
    : resolution(resolution), xDomain(xDomain), yDomain(yDomain) {
    backgroundColors.resize(resolution, std::vector<ImU32>(resolution));
}

ImU32 HeatMap::getColor(double value, bool opaque) {
    // Clamp value to [-1, 1]
    value = std::max(-1.0, std::min(1.0, value));

    ImVec4 color_blue(0.031f, 0.467f, 0.741f, 1.0f);   // #0877bd
    ImVec4 color_gray(0.910f, 0.918f, 0.922f, 1.0f);   // #e8eaeb
    ImVec4 color_orange(0.961f, 0.576f, 0.133f, 1.0f); // #f59322

    ImVec4 final_color;
    if (value < 0) {
        // Interpolate between blue (at -1) and gray (at 0)
        float t = static_cast<float>(value + 1.0);
        final_color.x = color_blue.x + t * (color_gray.x - color_blue.x);
        final_color.y = color_blue.y + t * (color_gray.y - color_blue.y);
        final_color.z = color_blue.z + t * (color_gray.z - color_blue.z);
    } else {
        // Interpolate between gray (at 0) and orange (at 1)
        float t = static_cast<float>(value);
        final_color.x = color_gray.x + t * (color_orange.x - color_gray.x);
        final_color.y = color_gray.y + t * (color_orange.y - color_gray.y);
        final_color.z = color_gray.z + t * (color_orange.z - color_gray.z);
    }

    // Set alpha based on the opaque flag
    final_color.w = opaque ? 1.0f : 0.627f; // Set alpha based on the flag
    ImU32 result_color = ImGui::ColorConvertFloat4ToU32(final_color);
  ///  std::cout << "  getColor for value " << value << " (opaque: " << opaque << ") -> R:" << ((result_color >> IM_COL32_R_SHIFT) & 0xFF) << " G:" << ((result_color >> IM_COL32_G_SHIFT) & 0xFF) << " B:" << ((result_color >> IM_COL32_B_SHIFT) & 0xFF) << " A:" << ((result_color >> IM_COL32_A_SHIFT) & 0xFF) << std::endl;
    return result_color;
}

// updateBackground uses the default semi-transparent color
void HeatMap::updateBackground(const std::vector<std::vector<double>>& data, bool discretize) {
    if (data.empty() || data.size() != resolution || data[0].size() != resolution) {
        return;
    }
    for (int i = 0; i < resolution; ++i) {
        for (int j = 0; j < resolution; ++j) {
            double value = data[i][j];
            if (discretize) {
                value = (value >= 0) ? 1.0 : -1.0;
            }
            backgroundColors[i][j] = getColor(value,false); // opaque = false by default
        }
    }
}

ImVec2 HeatMap::scale(double x, double y, ImVec2 p0, ImVec2 p1) {
    float screen_x = static_cast<float>(map_range(x, xDomain.first, xDomain.second, p0.x, p1.x));
    float screen_y = static_cast<float>(map_range(y, yDomain.first, yDomain.second, p0.y, p1.y)); // Y is inverted
   // std::cout << "  Scaling (" << x << ", " << y << ") from domain [" << xDomain.first << "," << xDomain.second << "]x[" << yDomain.first << "," << yDomain.second << "] to screen (" << screen_x << ", " << screen_y << ")" << std::endl;
    return ImVec2(screen_x, screen_y);
}

// THE FINAL DRAW FUNCTION
void HeatMap::draw(ImDrawList* drawList, ImVec2 canvas_p0, ImVec2 canvas_sz) {
    ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
    float cell_w = canvas_sz.x / resolution;
    float cell_h = canvas_sz.y / resolution;

    // 1. Draw background heatmap
    for (int i = 0; i < resolution; ++i) {
        for (int j = 0; j < resolution; ++j) {
            ImVec2 cell_p0 = ImVec2(canvas_p0.x + i * cell_w, canvas_p0.y + j * cell_h);
            ImVec2 cell_p1 = ImVec2(cell_p0.x + cell_w, cell_p0.y + cell_h);
            drawList->AddRectFilled(cell_p0, cell_p1, backgroundColors[i][j]);
        }
    }
}

void HeatMap::drawDataPoints(ImDrawList* drawList, ImVec2 canvas_p0, ImVec2 canvas_sz, const std::vector<playground::Example2D>& dataPoints) {
    ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
    float pointRadius = 4.5f;

    //std::cout << "HeatMap::drawDataPoints called with " << dataPoints.size() << " data points." << std::endl;

    for (const auto& point : dataPoints) {
        ImVec2 screenPos = scale(point.x, point.y, canvas_p0, canvas_p1);
        ImU32 color = getColor(point.label, true); // Use opaque color for data points
        drawList->AddCircleFilled(screenPos, pointRadius, color);
       // std::cout << "  Drawing point at (" << point.x << ", " << point.y << ") -> screen (" << screenPos.x << ", " << screenPos.y << ") with label " << point.label << std::endl;
    }
}