# Standalone benchmarks for microsoft/STL#6404 (ranges::min/max/minmax double evaluation).
# Reuses the google-benchmark submodule and the compiler flags of the STL's own benchmark suite,
# so numbers are comparable with benchmarks/src/minmax_element.cpp upstream.

cmake_minimum_required(VERSION 4.3.1)
project(minmax_benchmarks LANGUAGES CXX)

set(STL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../STL" CACHE PATH
    "Checkout of microsoft/STL; supplies benchmarks/google-benchmark.")
cmake_path(ABSOLUTE_PATH STL_SOURCE_DIR NORMALIZE)

set(GOOGLE_BENCHMARK_DIR "${STL_SOURCE_DIR}/benchmarks/google-benchmark")
if(NOT EXISTS "${GOOGLE_BENCHMARK_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "google-benchmark not found under '${GOOGLE_BENCHMARK_DIR}'.\n"
        "Run: git -C \"${STL_SOURCE_DIR}\" submodule update --init --depth 1 benchmarks/google-benchmark")
endif()

# Optionally measure against a locally built STL instead of the installed one, so that the
# _Prefer_iterator_copies heuristic in <xutility>/<algorithm> can be edited and A/B tested.
# Configure with -DSTL_BINARY_DIR=D:/Code/STL/out/build/x64 (a dir containing out/inc and out/lib).
if(DEFINED STL_BINARY_DIR)
    cmake_path(ABSOLUTE_PATH STL_BINARY_DIR NORMALIZE)
    if(NOT EXISTS "${STL_BINARY_DIR}/out")
        message(FATAL_ERROR "Invalid STL_BINARY_DIR '${STL_BINARY_DIR}'")
    endif()
    include_directories(BEFORE "${STL_BINARY_DIR}/out/inc")
    link_directories(BEFORE "${STL_BINARY_DIR}/out/lib/amd64")
    message(STATUS "Measuring the STL built in '${STL_BINARY_DIR}'")
else()
    message(STATUS "STL_BINARY_DIR not set; measuring the globally installed standard library")
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
set(CMAKE_BUILD_TYPE Release) # /O2 /Ob2, plus /Zi + /DEBUG below to keep profiling possible

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/Zi;/nologo;/diagnostics:caret;/W4;/WX;/w14265;/w15038;/w15262;/utf-8>")
add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:preprocessor>")
add_link_options("/DEBUG")

set(BENCHMARK_ENABLE_DOXYGEN OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory("${GOOGLE_BENCHMARK_DIR}" google-benchmark EXCLUDE_FROM_ALL)

# A benchmark target built against google-benchmark.
function(add_bench name)
    add_executable(bench-${name} ${ARGN})
    target_compile_features(bench-${name} PRIVATE cxx_std_23)
    target_include_directories(bench-${name} PRIVATE inc "${STL_SOURCE_DIR}/benchmarks/inc")
    target_link_libraries(bench-${name} PRIVATE benchmark::benchmark)
endfunction()

# A plain diagnostic executable, no google-benchmark.
function(add_tool name)
    add_executable(${name} ${ARGN})
    target_compile_features(${name} PRIVATE cxx_std_23)
    target_include_directories(${name} PRIVATE inc)
endfunction()

# Like add_tool, but also emits an annotated assembly listing (.cod) into the build tree.
# Several conclusions in the README rest on reading these rather than on timings.
function(add_asm_tool name)
    add_tool(${name} ${ARGN})
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/FAcs>)
endfunction()

add_tool(probe probe/probe.cpp)
add_tool(verdicts probe/verdicts.cpp)
add_tool(iterators probe/iterators.cpp)
add_tool(zip_types probe/zip_types.cpp)

add_asm_tool(asm_probe probe/asm_probe.cpp)
add_asm_tool(asm_real probe/asm_real.cpp)

add_bench(size_map src/size_map.cpp)
add_bench(scenarios src/scenarios.cpp)
add_bench(summary src/summary.cpp)
add_bench(ranges_minmax upstream/ranges_minmax.cpp)
