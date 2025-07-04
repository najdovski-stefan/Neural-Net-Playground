cmake_minimum_required(VERSION 3.16)

project(PlaygroundApp VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(glfw3_DIR /usr/local/Cellar/glfw/3.4/lib/cmake/glfw3)
# Find GLFW3 package (rely on system or Homebrew's CMake config)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# ImGui library
add_library(imgui_lib
    vendor/imgui/imgui.cpp
    vendor/imgui/imgui_draw.cpp
    vendor/imgui/imgui_tables.cpp
    vendor/imgui/imgui_widgets.cpp
    vendor/imgui/backends/imgui_impl_glfw.cpp
    vendor/imgui/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    vendor/imgui
    vendor/imgui/backends
    /usr/local/Cellar/glfw/3.4/include
)

# ImPlot library
add_library(implot_lib
    vendor/implot/implot.cpp
    vendor/implot/implot_items.cpp
)
target_include_directories(implot_lib PUBLIC vendor/implot)
target_link_libraries(implot_lib PUBLIC imgui_lib)

set(APP_SOURCES
    src/main.cpp
    src/dataset.cpp
    src/nn.cpp
    src/heatmap.cpp
    src/linechart.cpp
    src/playground.cpp
        vendor/glad/glad.c
)

add_executable(playground_app ${APP_SOURCES})

target_compile_definitions(playground_app PRIVATE IMGUI_HAS_DOCK)

target_include_directories(playground_app PRIVATE
    src
    vendor
    vendor/glad
    /usr/local/Cellar/glfw/3.4/include
)

target_link_libraries(playground_app PRIVATE
    implot_lib
    /usr/local/Cellar/glfw/3.4/lib/libglfw3.a
    OpenGL::GL
)

if(APPLE)
    target_link_libraries(playground_app PRIVATE
        "-framework CoreFoundation"
        "-framework IOKit"
        "-framework Cocoa"
        "-framework CoreVideo"
    )
endif()

install(TARGETS playground_app DESTINATION bin)

# ==============================================================================
# TESTING
# ==============================================================================
enable_testing()

add_executable(test_nn src/test_nn.cpp src/nn.cpp)
target_include_directories(test_nn PRIVATE src)
add_test(NAME test_nn COMMAND test_nn)

add_executable(test_dataset src/test_dataset.cpp src/dataset.cpp)
target_include_directories(test_dataset PRIVATE src)
add_test(NAME test_dataset COMMAND test_dataset)

add_executable(test_feature src/test_feature.cpp src/playground.cpp src/dataset.cpp src/nn.cpp src/heatmap.cpp src/linechart.cpp)
target_include_directories(test_feature PRIVATE src vendor vendor/glad)
target_link_libraries(test_feature PRIVATE implot_lib)
add_test(NAME test_feature COMMAND test_feature)
