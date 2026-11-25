cmake_minimum_required(VERSION 3.16)
project(cpp-yolo26n LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug or Release)" FORCE)
endif()

option(ENABLE_NATIVE_ARCH "Enable -march=native -mtune=native for Release builds" ON)

set(ORT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "Path to ONNX Runtime headers")
set(ORT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib" CACHE PATH "Path to libonnxruntime.so")

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ORT_LIB_DIR}" /usr/lib /usr/local/lib
    REQUIRED
)

add_library(yolo_core STATIC
    src/detector.cpp
    src/image.cpp
    src/preprocessing.cpp
    src/postprocessing.cpp
    src/benchmark.cpp
    src/metrics.cpp
    src/stb_impl.c
)

target_include_directories(yolo_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${ORT_INCLUDE_DIR}"
)

target_link_libraries(yolo_core PUBLIC "${ONNXRUNTIME_LIB}" m)

target_compile_options(yolo_core PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<CONFIG:Release>,$<BOOL:${ENABLE_NATIVE_ARCH}>>:-march=native -mtune=native>
)

add_executable(cpp-yolo26n src/main.cpp)
target_link_libraries(cpp-yolo26n PRIVATE yolo_core)
set_target_properties(cpp-yolo26n PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR};/usr/lib"
    INSTALL_RPATH "$ORIGIN/../../lib;$ORIGIN/../lib;/usr/lib"
)

enable_testing()
add_executable(cpp-yolo26n-tests src/unit_tests.cpp)
target_link_libraries(cpp-yolo26n-tests PRIVATE yolo_core)
add_test(NAME cpp_unit_tests COMMAND cpp-yolo26n-tests)
