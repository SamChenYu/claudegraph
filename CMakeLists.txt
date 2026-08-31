cmake_minimum_required(VERSION 3.12)
project(convograph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- FTXUI (vendored as a git submodule) -----------------------------------
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/FTXUI/CMakeLists.txt")
  message(FATAL_ERROR
    "external/FTXUI is empty. Run: git submodule update --init --recursive")
endif()

set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
add_subdirectory(external/FTXUI EXCLUDE_FROM_ALL)

# --- convograph ------------------------------------------------------------
add_executable(convograph src/convograph.cpp)
target_link_libraries(convograph
  PRIVATE
    ftxui::component
    ftxui::dom
    ftxui::screen
)
